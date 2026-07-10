"""
Recipe RAG pipeline: query understanding -> filtered retrieval -> generation.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load env BEFORE importing langfuse (so the SDK reads credentials at init) and
# before the langfuse-wrapped openai. Both are best practices from the langfuse skill.
load_dotenv(Path(__file__).parent.parent / ".env")

import instructor
from langfuse import observe
from langfuse.openai import openai  # drop-in replacement: auto-traces every OpenAI call
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from flashrank import Ranker, RerankRequest

from ragu.prompt_loader import render_prompt

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_COLLECTION = "Recipes-collection-01-hybrid"

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
ranker = Ranker(model_name=RERANK_MODEL)

def get_embedding(text, model="text-embedding-3-small"):
    # langfuse.openai auto-logs an embedding generation with model + token usage;
    # `name` just labels it in the trace. No manual usage bookkeeping needed.
    response = openai.embeddings.create(input=text, model=model, name="embed_query")
    return response.data[0].embedding


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


extractor_client = instructor.from_provider(
    "openai/gpt-5.4-nano",
    mode=instructor.Mode.RESPONSES_TOOLS,
)

@observe(name="extract_constraints")
def extract_constraints(query: str) -> ExtractedConstraints:
    # instructor builds its client from the langfuse-wrapped openai, so the
    # underlying Responses call is auto-traced as a nested generation (model +
    # tokens captured once). We just wrap it in a clearly-named span.
    return extractor_client.create(
        messages=[
            {"role": "system", "content": render_prompt(PROMPTS_DIR, "extract_constraints")},
            {"role": "user", "content": query},
        ],
        reasoning={"effort": "none"},
        response_model=ExtractedConstraints,
    )


def build_qdrant_filter(constraints: list[Constraint]) -> models.Filter | None:
    if not constraints:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=c.field, range=models.Range(**{c.op: c.value}))
            for c in constraints
        ]
    )


def query_points_with_filter(query, qfilter, k=5, collection_name=DEFAULT_COLLECTION, hybrid=True):
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


def query_points_with_fallback(query, k=5, collection_name=DEFAULT_COLLECTION, hybrid=True):
    extracted = extract_constraints(query)
    qfilter = build_qdrant_filter(extracted.constraints)

    results = query_points_with_filter(query, qfilter, k, collection_name, hybrid)
    filter_relaxed = False
    if qfilter is not None and not results.points:
        results = query_points_with_filter(query, None, k, collection_name, hybrid)
        filter_relaxed = True

    return {
        "results": results,
        "constraints": extracted.constraints,
        "filter_relaxed": filter_relaxed,
    }


@observe(as_type="retriever", name="retrieve_data")
def retrieve_data(query, k=5, collection_name=DEFAULT_COLLECTION, hybrid=True):
    out = query_points_with_fallback(query, k, collection_name, hybrid)
    recipes = []
    for result in out["results"].points:
        payload = result.payload
        images = payload.get("Images") or []
        recipes.append({
            "id": int(payload["RecipeId"]),
            "name": payload["Name"],
            "description": payload.get("Description"),
            "text": payload.get("text"),
            "image": images[0] if images else None,
            "category": payload.get("RecipeCategory"),
            "keywords": payload.get("Keywords") or [],
            "score": result.score,
            "rating": payload["bayesian_rating"],
            "n_ratings": payload["n_ratings"],
            "calories": payload.get("Calories"),
            "protein": payload.get("ProteinContent"),
            "carbs": payload.get("CarbohydrateContent"),
            "fat": payload.get("FatContent"),
            "total_time": payload.get("total_time_minutes"),
            "ingredients": payload.get("RecipeIngredientParts") or [],
            "instructions": payload.get("RecipeInstructions") or [],
        })
    return {
        "recipes": recipes,
        "constraints": [c.model_dump() for c in out["constraints"]],
        "filter_relaxed": out["filter_relaxed"],
    }


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


@observe(name="build_prompt")
def build_system_prompt(context, constraints_not_satisfied=False):
    return render_prompt(
        PROMPTS_DIR,
        "system_prompt",
        context=context,
        constraints_not_satisfied=constraints_not_satisfied,
    )


@observe(as_type="retriever", name="rerank")
def rerank(query, recipes, top_n=5):
    if not recipes:  # ponytail: Cohere 400s on an empty/all-blank document list
        return []
    docs = [{"id": i, "text": r.get("text")} for i, r in enumerate(recipes)]
    response = ranker.rerank(RerankRequest(query=query, passages=docs))
    out = []
    for result in response[:top_n]:
        recipe = recipes[result["id"]]
        recipe["score"] = result["score"]
        out.append(recipe)
    return out


def generate_answer(system_prompt, question):
    # drop-in auto-captures model + token usage; `name` labels the generation.
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        reasoning_effort="none",
        name="generate_answer",
    )
    return response.choices[0].message.content


@observe(name="rag_pipeline")
def rag_pipeline(question, k=5, candidates=20, collection_name=DEFAULT_COLLECTION, hybrid=True, use_rerank=True):
    retrieved = retrieve_data(question, candidates, collection_name, hybrid=hybrid)
    filter_relaxed = retrieved["filter_relaxed"]
    # rerank picks + reorders the best k of `candidates`; without it, fusion top-k as-is
    recipes = rerank(question, retrieved["recipes"], top_n=k) if use_rerank else retrieved["recipes"][:k]
    blocks = format_blocks(recipes)
    system_prompt = build_system_prompt("\n\n".join(blocks), constraints_not_satisfied=filter_relaxed)
    answer = generate_answer(system_prompt, question)
    return {
        "question": question,
        "answer": answer,
        "recipes": recipes,
        "constraints": retrieved["constraints"],
        "filter_relaxed": filter_relaxed,
        "retrieved_context_ids": [r["id"] for r in recipes],
        "retrieved_context": blocks,
        "retrieved_payloads": [
            {
                "Calories": r["calories"],
                "ProteinContent": r["protein"],
                "CarbohydrateContent": r["carbs"],
                "FatContent": r["fat"],
                "total_time_minutes": r["total_time"],
            }
            for r in recipes
        ],
    }
