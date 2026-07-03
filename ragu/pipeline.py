"""
Recipe RAG pipeline: query understanding -> filtered retrieval -> generation.
"""

import os
from pathlib import Path
from typing import Literal

import cohere
import instructor
import openai
from dotenv import load_dotenv
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from ragu.prompt_loader import render_prompt

load_dotenv(Path(__file__).parent.parent / ".env")

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_COLLECTION = "Recipes-collection-01-hybrid"

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
cohere_client = cohere.ClientV2()

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
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

@traceable(
    name="extract_constraints",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-5.4-nano"},
)
def extract_constraints(query: str) -> ExtractedConstraints:
    result, completion = extractor_client.create_with_completion(
        messages=[
            {"role": "system", "content": render_prompt(PROMPTS_DIR, "extract_constraints")},
            {"role": "user", "content": query},
        ],
        reasoning={"effort": "none"},
        response_model=ExtractedConstraints,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    return result


def build_qdrant_filter(constraints: list[Constraint]) -> models.Filter | None:
    if not constraints:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=c.field, range=models.Range(**{c.op: c.value}))
            for c in constraints
        ]
    )


def query_points_with_filter(query, qfilter, k=5, collection_name=DEFAULT_COLLECTION):
    query_embedding = get_embedding(query)
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


def query_points_with_fallback(query, k=5, collection_name=DEFAULT_COLLECTION):
    extracted = extract_constraints(query)
    qfilter = build_qdrant_filter(extracted.constraints)

    results = query_points_with_filter(query, qfilter, k, collection_name)
    filter_relaxed = False
    if qfilter is not None and not results.points:
        results = query_points_with_filter(query, None, k, collection_name)
        filter_relaxed = True

    return {
        "results": results,
        "constraints": extracted.constraints,
        "filter_relaxed": filter_relaxed,
    }


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, k=5, collection_name=DEFAULT_COLLECTION):
    out = query_points_with_fallback(query, k, collection_name)
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


@traceable(name="format_retrieved_context", run_type="prompt")
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


@traceable(name="build_prompt", run_type="prompt")
def build_system_prompt(context, constraints_not_satisfied=False):
    return render_prompt(
        PROMPTS_DIR,
        "system_prompt",
        context=context,
        constraints_not_satisfied=constraints_not_satisfied,
    )


@traceable(name="rerank", run_type="retriever")
def rerank(query, recipes, top_n=5):
    if not recipes:  # ponytail: Cohere 400s on an empty/all-blank document list
        return []
    docs = [r.get("text") for r in recipes]
    response = cohere_client.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=docs,
        top_n=top_n,
    )
    out = []
    for result in response.results:
        recipe = recipes[result.index]
        recipe["score"] = result.relevance_score # replace fusion score with rerank score
        out.append(recipe)
    return out


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-5.4-nano"},
)
def generate_answer(system_prompt, question):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        reasoning_effort="none",
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.choices[0].message.content


@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline(question, k=5, candidates=20, collection_name=DEFAULT_COLLECTION):
    retrieved = retrieve_data(question, candidates, collection_name)
    filter_relaxed = retrieved["filter_relaxed"]
    recipes = rerank(question, retrieved["recipes"], top_n=k)
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
