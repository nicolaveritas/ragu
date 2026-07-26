"""Recipe retrieval: query understanding -> filtered hybrid search -> payload shaping.

Public API (bottom of file):
    retrieve_data        - hybrid search with numeric-constraint filtering + fallback
    retrieve_and_rerank  - retrieve_data + rerank, the default search config
    fetch_recipes_by_ids - full recipe cards for known ids (HTTP card lookups)
    format_blocks        - render recipe dicts into the text blocks the LLM reads

Reranking is generic and lives in _shared.rerank. This module owns everything
recipe-specific: the constraint model, the Qdrant query building, payload shaping.
Ordered definition-before-use: models, then _private helpers, then public API.
"""

from typing import Literal

import instructor
import yaml
from langfuse import observe
from pydantic import BaseModel, Field
from qdrant_client import models

from ricettario.core._shared import PROMPTS_DIR, get_embedding, qdrant_client, rerank

RECIPES_COLLECTION = "Recipes-collection-01-hybrid"

# One prompt, no template variables: read it directly rather than import ragu's
# Jinja loader — nothing in ricettario imports from ragu.
EXTRACT_CONSTRAINTS_PROMPT = yaml.safe_load(
    (PROMPTS_DIR / "extract_constraints.yaml").read_text()
)["template"].strip()


# ============================ Query-understanding model =======================


class Constraint(BaseModel):
    field: Literal['Calories', 'ProteinContent', 'CarbohydrateContent', 'FatContent', 'total_time_minutes'] = Field(
        description=(
            "Recipe attribute the constraint applies to. Units are fixed: "
            "Calories is kcal; ProteinContent, CarbohydrateContent and FatContent are grams; "
            "total_time_minutes is minutes."
        )
    )
    op: Literal['gt', 'gte', 'lt', 'lte'] = Field(
        description=(
            "Comparison operator. 'more than' -> gt, 'at least' / 'or more' -> gte, "
            "'less than' / 'under' -> lt, 'at most' / 'or less' / 'max' -> lte."
        )
    )
    value: float = Field(
        description=(
            "Threshold, converted to the field's unit when the query uses a different one "
            "(e.g. 'ready in 2 hours' -> 120 for total_time_minutes)."
        )
    )


class ExtractedConstraints(BaseModel):
    reasoning: str = Field(
        description="Short analysis of which numeric thresholds, if any, the query explicitly states."
    )
    constraints: list[Constraint] = Field(
        description="Numeric constraints explicitly stated in the query. Empty if it states none."
    )


# =============================== Private helpers ==============================


_extractor_client = instructor.from_provider(
    "openai/gpt-5.4-nano",
    mode=instructor.Mode.RESPONSES_TOOLS,
)


@observe(name="extract_constraints")
def _extract_constraints(query: str) -> ExtractedConstraints:
    # instructor builds its client from the langfuse-wrapped openai, so the
    # underlying Responses call is auto-traced as a nested generation (model +
    # tokens captured once). We just wrap it in a clearly-named span.
    return _extractor_client.create(
        messages=[
            {"role": "system", "content": EXTRACT_CONSTRAINTS_PROMPT},
            {"role": "user", "content": query},
        ],
        reasoning={"effort": "none"},
        response_model=ExtractedConstraints,
    )


def _build_qdrant_filter(constraints: list[Constraint]) -> models.Filter | None:
    if not constraints:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=c.field, range=models.Range(**{c.op: c.value}))
            for c in constraints
        ]
    )


@observe(name="qdrant_search")
def _query_points_with_filter(query, qfilter, k=5, collection_name=RECIPES_COLLECTION, hybrid=True):
    query_embedding = get_embedding(query)
    if not hybrid:  # dense-only: single vector query, no BM25 prefetch, no fusion
        return qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="text-embedding-3-small",
            query_filter=qfilter,
            limit=k,
        )
    return qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_embedding,
                filter=qfilter,
                using="text-embedding-3-small",
                limit=20
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                filter=qfilter,
                using="bm25",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion="rrf"),
        limit=k
    )


def _query_points_with_fallback(query, k=5, collection_name=RECIPES_COLLECTION, hybrid=True):
    extracted = _extract_constraints(query)
    qfilter = _build_qdrant_filter(extracted.constraints)

    results = _query_points_with_filter(query, qfilter, k, collection_name, hybrid)
    filter_relaxed = False
    if qfilter is not None and not results.points:
        results = _query_points_with_filter(query, None, k, collection_name, hybrid)
        filter_relaxed = True

    return {
        "results": results,
        "constraints": extracted.constraints,
        "filter_relaxed": filter_relaxed,
    }


def _payload_to_recipe(payload, score=None):
    images = payload.get("Images") or []
    return {
        "id": int(payload["RecipeId"]),
        "name": payload["Name"],
        "description": payload.get("Description"),
        "text": payload.get("text"),
        "image": images[0] if images else None,
        "category": payload.get("RecipeCategory"),
        "keywords": payload.get("Keywords") or [],
        "score": score,
        "rating": payload["bayesian_rating"],
        "n_ratings": payload["n_ratings"],
        "calories": payload.get("Calories"),
        "protein": payload.get("ProteinContent"),
        "carbs": payload.get("CarbohydrateContent"),
        "fat": payload.get("FatContent"),
        "total_time": payload.get("total_time_minutes"),
        "ingredients": payload.get("RecipeIngredientParts") or [],
        "instructions": payload.get("RecipeInstructions") or [],
    }


# ================================= Public API =================================


@observe(as_type="retriever", name="retrieve_data")
def retrieve_data(query, k=5, collection_name=RECIPES_COLLECTION, hybrid=True):
    out = _query_points_with_fallback(query, k, collection_name, hybrid)
    recipes = [_payload_to_recipe(r.payload, score=r.score) for r in out["results"].points]
    return {
        "recipes": recipes,
        "constraints": [c.model_dump() for c in out["constraints"]],
        "filter_relaxed": out["filter_relaxed"],
    }


@observe(name="retrieve_and_rerank")
def retrieve_and_rerank(query, top_k=5, candidates=20):
    """The default recipe search: wide fusion recall, then rerank down to top_k."""
    retrieved = retrieve_data(query, k=candidates)
    return {
        "recipes": rerank(query, retrieved["recipes"], top_n=top_k),
        "filter_relaxed": retrieved["filter_relaxed"],
    }


def fetch_recipes_by_ids(ids, collection_name=RECIPES_COLLECTION):
    """Full card payloads for recipe ids, in the given order (missing ids skipped).

    Point id == RecipeId (see notebook 09), so a single retrieve() suffices.
    """
    if not ids:
        return []
    records = qdrant_client.retrieve(collection_name=collection_name, ids=ids, with_payload=True)
    by_id = {int(r.payload["RecipeId"]): _payload_to_recipe(r.payload) for r in records}
    return [by_id[i] for i in ids if i in by_id]


@observe(name="format_retrieved_context")
def format_blocks(retrieved, max_steps_chars=300):
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        nutrition = []
        if r["calories"] is not None:
            nutrition.append(f"{round(r['calories'])} kcal")
        if r["protein"] is not None:
            nutrition.append(f"P {round(r['protein'])}g")
        if r["carbs"] is not None:
            nutrition.append(f"C {round(r['carbs'])}g")
        if r["fat"] is not None:
            nutrition.append(f"F {round(r['fat'])}g")

        ingredients = ", ".join(str(x) for x in r["ingredients"] if x)
        steps = " ".join(str(s) for s in r["instructions"] if s)
        if len(steps) > max_steps_chars:
            steps = steps[:max_steps_chars].rstrip() + "..."
        tt = r["total_time"]

        desc = (r.get("description") or "").strip()
        desc_line = f"  description: {desc}\n" if desc else ""

        blocks.append(
            f"[{i}] {r['name']} (id: {r['id']})\n"
            f"{desc_line}"
            f"  rating: {r['rating']:.1f} ({r['n_ratings']} reviews)\n"
            f"  nutrition: {' | '.join(nutrition) or 'n/a'}\n"
            f"  total time: {f'{tt} min' if tt is not None else 'n/a'}\n"
            f"  ingredients: {ingredients}\n"
            f"  steps: {steps}"
        )
    return blocks
