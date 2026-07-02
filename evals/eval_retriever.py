"""Run the RAG eval experiment on LangSmith.

Scores the recipe RAG pipeline against the eval dataset built in
`notebooks/06-rag-eval-dataset.ipynb`.

- Retrieval (deterministic): context_recall, hit_at_k, mrr,
  constraint_satisfaction.
- Generation (RAGAS, LLM-judged): faithfulness, response_relevancy.

Evaluators return None where a metric doesn't apply, and LangSmith skips them.
Metric definitions: docs/evals-cheatsheet.md.

Retrieval runs against the 100-recipe eval sample collection so every
ground-truth recipe is guaranteed to be in the index.

Usage (from the repo root, with Qdrant running on localhost:6333):

    uv run python evals/eval_retriever.py            # full run
    uv run python evals/eval_retriever.py --smoke    # 3-example smoke test
"""

import argparse
import asyncio
from pathlib import Path

import openai
from dotenv import load_dotenv
from langsmith import Client
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

load_dotenv(Path(__file__).parent.parent / ".env")

EVAL_COLLECTION_NAME = "Recipes-collection-01-eval-sample-100"
DATASET_NAME = "ragu-evaluation-dataset"

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

async_openai_client = AsyncOpenAI()
ragas_llm = llm_factory("gpt-5.4-mini", client=async_openai_client)
# ragas doesn't recognize dotted model versions ("gpt-5.4") as reasoning
# models and would send params the API rejects; set them ourselves.
ragas_llm.model_args = {"max_completion_tokens": 4096, "temperature": 1.0}
ragas_embeddings = RagasOpenAIEmbeddings(
    client=async_openai_client,
    model="text-embedding-3-small",
)
faithfulness_scorer = Faithfulness(llm=ragas_llm)
relevancy_scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)


# --- RAG pipeline (same as notebooks/04-rag-pipeline.ipynb) ------------------


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=EVAL_COLLECTION_NAME,
        query=query_embedding,
        limit=k,
    )
    retrieved = []
    for result in results.points:
        payload = result.payload
        retrieved.append({
            "id": int(payload["RecipeId"]),
            "name": payload["Name"],
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
    return retrieved


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

        blocks.append(
            f"[{i}] {r['name']} (id: {r['id']})\n"
            f"  rating: {r['rating']:.1f} ({r['n_ratings']} reviews)\n"
            f"  nutrition: {' | '.join(nutrition) or 'n/a'}\n"
            f"  total time: {f'{tt} min' if tt is not None else 'n/a'}\n"
            f"  ingredients: {ingredients}\n"
            f"  steps: {steps}"
        )
    return blocks


def build_system_prompt(context):
    return f"""
You are a helpful cooking assistant. You help people decide what to cook by recommending recipes from the ones available below.

Instructions:
- Only recommend recipes from the available recipes. Never invent recipes, ingredients, or nutrition values.
- Refer to recipes by their name; you may add the id in parentheses so it can be looked up.
- If the question has constraints (calories, time, an ingredient to include or avoid, a meal type), respect them and prefer recipes that match.
- If none of the available recipes fit the request well, say so honestly instead of forcing a poor match.
- The steps shown are only a short preview, not the full method, so don't present them as complete instructions.
- Keep the answer concise and friendly. Do not use markdown.

Available recipes:
{context}
"""


def generate_answer(system_prompt, question):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        reasoning_effort="none",
    )
    return response.choices[0].message.content


def rag_pipeline(question, k=5):
    retrieved = retrieve_data(question, k)
    blocks = format_blocks(retrieved)
    answer = generate_answer(build_system_prompt("\n\n".join(blocks)), question)
    return {
        "question": question,
        "answer": answer,
        "retrieved_context_ids": [r["id"] for r in retrieved],
        "retrieved_context": blocks,
        # Keys must match the constraint "field" names in the dataset.
        "retrieved_payloads": [
            {
                "Calories": r["calories"],
                "ProteinContent": r["protein"],
                "total_time_minutes": r["total_time"],
            }
            for r in retrieved
        ],
    }


# --- Retrieval evaluators -----------------------------------------------------
# LangSmith calls each evaluator with the traced run (run.outputs = what
# rag_pipeline returned) and the dataset example (example.outputs = ground truth).


def _reference_ids(example):
    return [str(x) for x in example.outputs.get("reference_context_ids") or []]


def _retrieved_ids(run):
    return [str(x) for x in run.outputs["retrieved_context_ids"]]


def eval_context_recall(run, example):
    reference = set(_reference_ids(example))
    if not reference:
        return {"key": "context_recall", "score": None}
    retrieved = set(_retrieved_ids(run))
    return {"key": "context_recall", "score": len(reference & retrieved) / len(reference)}


def eval_hit_at_k(run, example):
    reference = set(_reference_ids(example))
    if not reference:
        return {"key": "hit_at_k", "score": None}
    hit = any(rid in reference for rid in _retrieved_ids(run))
    return {"key": "hit_at_k", "score": 1.0 if hit else 0.0}


def eval_mrr(run, example):
    reference = set(_reference_ids(example))
    if not reference:
        return {"key": "mrr", "score": None}
    for rank, rid in enumerate(_retrieved_ids(run), start=1):
        if rid in reference:
            return {"key": "mrr", "score": 1.0 / rank}
    return {"key": "mrr", "score": 0.0}


_OPS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


def eval_constraint_satisfaction(run, example):
    """Fraction of retrieved recipes whose payload satisfies all constraints.

    A missing payload value counts as a violation (we can't verify it).
    """
    constraints = example.outputs.get("constraints") or []
    if not constraints:
        return {"key": "constraint_satisfaction", "score": None}
    payloads = run.outputs.get("retrieved_payloads") or []
    if not payloads:
        return {"key": "constraint_satisfaction", "score": 0.0}

    def satisfies(payload):
        for c in constraints:
            value = payload.get(c["field"])
            if value is None or not _OPS[c["op"]](value, c["value"]):
                return False
        return True

    score = sum(satisfies(p) for p in payloads) / len(payloads)
    return {"key": "constraint_satisfaction", "score": score}


# --- Generation evaluators (RAGAS) ---------------------------------------------


async def eval_faithfulness(run, example):
    result = await faithfulness_scorer.ascore(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    return {"key": "faithfulness", "score": result.value}


async def eval_response_relevancy(run, example):
    result = await relevancy_scorer.ascore(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
    )
    return {"key": "response_relevancy", "score": result.value}


# --- Experiment ----------------------------------------------------------------


async def run_experiment(smoke: bool):
    if smoke:
        dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)
        examples = list(ls_client.list_examples(dataset_id=dataset.id))
        # 2 gold-id examples + 1 constraint example exercise every evaluator.
        with_gold = [e for e in examples if (e.outputs or {}).get("reference_context_ids")]
        constraint = [e for e in examples if (e.outputs or {}).get("constraints")]
        data = with_gold[:2] + constraint[:1]
        experiment_prefix = "ragu-retriever-smoketest"
    else:
        data = DATASET_NAME
        experiment_prefix = "ragu-retriever"

    async def target(inputs: dict) -> dict:
        return rag_pipeline(inputs["question"])

    results = await ls_client.aevaluate(
        target,
        data=data,
        evaluators=[
            eval_context_recall,
            eval_hit_at_k,
            eval_mrr,
            eval_constraint_satisfaction,
            eval_faithfulness,
            eval_response_relevancy,
        ],
        experiment_prefix=experiment_prefix,
    )

    df = results.to_pandas()
    metric_cols = [c for c in df.columns if c.startswith("feedback.")]
    print("\nMean scores:")
    print(df[metric_cols].mean(numeric_only=True).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on 3 examples only")
    args = parser.parse_args()
    asyncio.run(run_experiment(smoke=args.smoke))


if __name__ == "__main__":
    main()
