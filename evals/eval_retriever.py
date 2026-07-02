"""Run the RAG eval experiment on LangSmith.

Scores the recipe RAG pipeline (ragu.pipeline) against the eval dataset built
in `notebooks/06-rag-eval-dataset.ipynb`.

- Retrieval (deterministic): context_recall, hit_at_k, mrr,
  constraint_satisfaction.
- Generation (RAGAS, LLM-judged): faithfulness, response_relevancy.

Evaluators return None where a metric doesn't apply, and LangSmith skips them.
Metric definitions: docs/evals.md.

Retrieval runs against the 100-recipe eval sample collection so every
ground-truth recipe is guaranteed to be in the index.

Usage (from the repo root, with Qdrant running on localhost:6333):

    uv run python evals/eval_retriever.py            # full run
    uv run python evals/eval_retriever.py --smoke    # 3-example smoke test
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from ragu.pipeline import rag_pipeline

load_dotenv(Path(__file__).parent.parent / ".env")

EVAL_COLLECTION_NAME = "Recipes-collection-01-eval-sample-100"
DATASET_NAME = "ragu-evaluation-dataset"

ls_client = Client()

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
        experiment_prefix = "ragu-retriever-filters"

    async def target(inputs: dict) -> dict:
        return rag_pipeline(inputs["question"], collection_name=EVAL_COLLECTION_NAME)

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
