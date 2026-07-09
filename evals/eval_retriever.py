"""Run the RAG eval experiment on LangSmith.

Scores the recipe RAG pipeline (ragu.pipeline) against the eval dataset built
in `notebooks/10-rag-eval-dataset-large.ipynb`.

- Retrieval (deterministic): context_recall, hit_at_k, mrr,
  constraint_satisfaction.
- Generation (RAGAS, LLM-judged): faithfulness, response_relevancy.
- Refusal (LLM-judged): correctly_declined, unanswerable bucket only.

Evaluators return None where a metric doesn't apply, and LangSmith skips them.
Metric definitions: docs/evals.md.

Retrieval runs against the FULL collection: gold labels are complete over the
whole corpus (relevance-swept + verified), so no sample collection is needed.

Usage (from the repo root, with Qdrant running on localhost:6333):

    uv run python evals/eval_retriever.py            # baseline: default config, all metrics
    uv run python evals/eval_retriever.py --sweep    # hybrid x rerank sweep, retrieval metrics only
    uv run python evals/eval_retriever.py --smoke    # 4-example smoke test (combines with --sweep)
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from langsmith import Client
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from pydantic import BaseModel

from ragu.pipeline import rag_pipeline
from ragu.prompt_loader import render_prompt

load_dotenv(Path(__file__).parent.parent / ".env")

PROMPTS_DIR = Path(__file__).parent / "prompts"
EVAL_COLLECTION_NAME = "Recipes-collection-01-hybrid"
DATASET_NAME = "ragu-evaluation-dataset-large"

DEFAULT_CONFIG = {"hybrid": True, "rerank": True, "k": 5, "candidates": 20}
# First sweep varies only the two booleans; k/candidates get their own sweep later
# (one variable family at a time - see plans/configurable-eval-sweep.md).
SWEEP_CONFIGS = {
    "dense": {**DEFAULT_CONFIG, "hybrid": False, "rerank": False},
    "dense-rerank": {**DEFAULT_CONFIG, "hybrid": False},
    "hybrid": {**DEFAULT_CONFIG, "rerank": False},
    "hybrid-rerank": DEFAULT_CONFIG,
}

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
    # RAGAS faithfulness punishes honest refusals ("no match, closest are...")
    # as unsupported claims, so it doesn't apply to the unanswerable bucket.
    if (example.metadata or {}).get("bucket") == "unanswerable":
        return {"key": "faithfulness", "score": None}
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


# --- Refusal evaluator (unanswerable bucket) -------------------------------------


class DeclineJudgment(BaseModel):
    reasoning: str
    declined: bool


async def eval_correctly_declined(run, example):
    if (example.metadata or {}).get("bucket") != "unanswerable":
        return {"key": "correctly_declined", "score": None}
    completion = await async_openai_client.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": render_prompt(PROMPTS_DIR, "decline_judge")},
            {
                "role": "user",
                "content": render_prompt(
                    PROMPTS_DIR,
                    "decline_judge_user",
                    question=run.outputs["question"],
                    answer=run.outputs["answer"],
                ),
            },
        ],
        response_format=DeclineJudgment,
    )
    judgment = completion.choices[0].message.parsed
    return {
        "key": "correctly_declined",
        "score": 1.0 if judgment.declined else 0.0,
        "comment": judgment.reasoning,
    }


# --- Experiment ----------------------------------------------------------------

# The sweep varies the *retriever*, so it runs only the retrieval metrics; the
# LLM-judged generation/refusal metrics mostly re-measure the generator (which
# doesn't change across configs) and only run in the single-config baseline.
RETRIEVAL_EVALUATORS = [eval_context_recall, eval_hit_at_k, eval_mrr, eval_constraint_satisfaction]
GENERATION_EVALUATORS = [eval_faithfulness, eval_response_relevancy, eval_correctly_declined]


async def run_one(name: str, cfg: dict, evaluators: list, data):
    async def target(inputs: dict) -> dict:
        return rag_pipeline(
            inputs["question"],
            k=cfg["k"],
            candidates=cfg["candidates"],
            collection_name=EVAL_COLLECTION_NAME,
            hybrid=cfg["hybrid"],
            use_rerank=cfg["rerank"],
        )

    results = await ls_client.aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix=f"ragu-{name}",
    )
    df = results.to_pandas()
    means = df[[c for c in df.columns if c.startswith("feedback.")]].mean(numeric_only=True)
    print(f"\n[{name}] mean scores:")
    print(means.to_string())
    return means


def smoke_examples():
    dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)
    examples = list(ls_client.list_examples(dataset_id=dataset.id))
    # 2 gold-id + 1 constraint + 1 unanswerable exercise every evaluator.
    with_gold = [e for e in examples if (e.outputs or {}).get("reference_context_ids")]
    constraint = [e for e in examples if (e.outputs or {}).get("constraints")]
    unanswerable = [e for e in examples if (e.metadata or {}).get("bucket") == "unanswerable"]
    return with_gold[:2] + constraint[:1] + unanswerable[:1]


async def run_experiment(smoke: bool, sweep: bool, config: str | None = None):
    data = smoke_examples() if smoke else DATASET_NAME
    suffix = "-smoke" if smoke else ""

    if config:
        await run_one(f"sweep-{config}{suffix}", SWEEP_CONFIGS[config], RETRIEVAL_EVALUATORS, data)
    elif sweep:
        means = {}
        for name, cfg in SWEEP_CONFIGS.items():
            means[name] = await run_one(f"sweep-{name}{suffix}", cfg, RETRIEVAL_EVALUATORS, data)
        print("\n=== sweep comparison ===")
        print(pd.DataFrame(means).T.to_string())
    else:
        await run_one(
            f"baseline{suffix}",
            DEFAULT_CONFIG,
            RETRIEVAL_EVALUATORS + GENERATION_EVALUATORS,
            data,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on 4 examples only")
    parser.add_argument("--sweep", action="store_true", help="hybrid x rerank config sweep, retrieval metrics only")
    parser.add_argument("--config", choices=list(SWEEP_CONFIGS), help="run a single sweep config (retrieval metrics only)")
    args = parser.parse_args()
    asyncio.run(run_experiment(smoke=args.smoke, sweep=args.sweep, config=args.config))


if __name__ == "__main__":
    main()
