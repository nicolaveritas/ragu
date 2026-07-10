"""Run the RAG eval experiment on Langfuse.

Scores the recipe RAG pipeline (ragu.pipeline) against the eval dataset in
Langfuse ("ragu-evaluation-dataset-large"). Upload/refresh the dataset first
with `evals/upload_dataset.py` (source of truth: data/eval_dataset_large.jsonl).

- Retrieval (deterministic): context_recall, hit_at_k, mrr,
  constraint_satisfaction.
- Generation (RAGAS, LLM-judged): faithfulness, response_relevancy.
- Refusal (LLM-judged): correctly_declined, unanswerable bucket only.

Evaluators return [] where a metric doesn't apply, and Langfuse records no score
for that item. Metric definitions: docs/evals.md.

Retrieval runs against the FULL collection: gold labels are complete over the
whole corpus (relevance-swept + verified), so no sample collection is needed.

Usage (from the repo root, with Qdrant + Langfuse running):

    uv run python evals/eval_retriever.py            # baseline: default config, all metrics
    uv run python evals/eval_retriever.py --sweep    # hybrid x rerank sweep, retrieval metrics only
    uv run python evals/eval_retriever.py --smoke    # 4-example smoke test (combines with --sweep)
"""

import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # env before langfuse import

from langfuse import Evaluation, get_client
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from ragu.pipeline import rag_pipeline, RERANK_MODEL
from ragu.prompt_loader import render_prompt

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

langfuse = get_client()

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
# Langfuse calls each evaluator with the task's return value (`output` = what
# rag_pipeline returned) and the dataset item (`expected_output` = ground truth,
# `metadata` = bucket/query_style). Return [] to record no score for this item.


def _reference_ids(expected_output):
    return [str(x) for x in (expected_output or {}).get("reference_context_ids") or []]


def _retrieved_ids(output):
    return [str(x) for x in output["retrieved_context_ids"]]


def eval_context_recall(*, output, expected_output, **kwargs):
    reference = set(_reference_ids(expected_output))
    if not reference:
        return []
    retrieved = set(_retrieved_ids(output))
    return Evaluation(name="context_recall", value=len(reference & retrieved) / len(reference))


def eval_hit_at_k(*, output, expected_output, **kwargs):
    reference = set(_reference_ids(expected_output))
    if not reference:
        return []
    hit = any(rid in reference for rid in _retrieved_ids(output))
    return Evaluation(name="hit_at_k", value=1.0 if hit else 0.0)


def eval_mrr(*, output, expected_output, **kwargs):
    reference = set(_reference_ids(expected_output))
    if not reference:
        return []
    for rank, rid in enumerate(_retrieved_ids(output), start=1):
        if rid in reference:
            return Evaluation(name="mrr", value=1.0 / rank)
    return Evaluation(name="mrr", value=0.0)


_OPS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


def eval_constraint_satisfaction(*, output, expected_output, **kwargs):
    """Fraction of retrieved recipes whose payload satisfies all constraints.

    A missing payload value counts as a violation (we can't verify it).
    """
    constraints = (expected_output or {}).get("constraints") or []
    if not constraints:
        return []
    payloads = output.get("retrieved_payloads") or []
    if not payloads:
        return Evaluation(name="constraint_satisfaction", value=0.0)

    def satisfies(payload):
        for c in constraints:
            value = payload.get(c["field"])
            if value is None or not _OPS[c["op"]](value, c["value"]):
                return False
        return True

    score = sum(satisfies(p) for p in payloads) / len(payloads)
    return Evaluation(name="constraint_satisfaction", value=score)


# --- Generation evaluators (RAGAS) ---------------------------------------------


async def eval_faithfulness(*, output, metadata, **kwargs):
    # RAGAS faithfulness punishes honest refusals ("no match, closest are...")
    # as unsupported claims, so it doesn't apply to the unanswerable bucket.
    if (metadata or {}).get("bucket") == "unanswerable":
        return []
    result = await faithfulness_scorer.ascore(
        user_input=output["question"],
        response=output["answer"],
        retrieved_contexts=output["retrieved_context"],
    )
    return Evaluation(name="faithfulness", value=result.value)


async def eval_response_relevancy(*, output, **kwargs):
    result = await relevancy_scorer.ascore(
        user_input=output["question"],
        response=output["answer"],
    )
    return Evaluation(name="response_relevancy", value=result.value)


# --- Refusal evaluator (unanswerable bucket) -------------------------------------


class DeclineJudgment(BaseModel):
    reasoning: str
    declined: bool


async def eval_correctly_declined(*, output, metadata, **kwargs):
    if (metadata or {}).get("bucket") != "unanswerable":
        return []
    completion = await async_openai_client.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": render_prompt(PROMPTS_DIR, "decline_judge")},
            {
                "role": "user",
                "content": render_prompt(
                    PROMPTS_DIR,
                    "decline_judge_user",
                    question=output["question"],
                    answer=output["answer"],
                ),
            },
        ],
        response_format=DeclineJudgment,
    )
    judgment = completion.choices[0].message.parsed
    return Evaluation(
        name="correctly_declined",
        value=1.0 if judgment.declined else 0.0,
        comment=judgment.reasoning,
    )


# --- Experiment ----------------------------------------------------------------

# The sweep varies the *retriever*, so it runs only the retrieval metrics; the
# LLM-judged generation/refusal metrics mostly re-measure the generator (which
# doesn't change across configs) and only run in the single-config baseline.
RETRIEVAL_EVALUATORS = [eval_context_recall, eval_hit_at_k, eval_mrr, eval_constraint_satisfaction]
GENERATION_EVALUATORS = [eval_faithfulness, eval_response_relevancy, eval_correctly_declined]


def make_task(cfg: dict):
    def task(*, item, **kwargs):
        # item is a DatasetItem (hosted run) or a plain dict (local smoke data)
        inp = item.input if hasattr(item, "input") else item["input"]
        return rag_pipeline(
            inp["question"],
            k=cfg["k"],
            candidates=cfg["candidates"],
            collection_name=EVAL_COLLECTION_NAME,
            hybrid=cfg["hybrid"],
            use_rerank=cfg["rerank"],
        )

    return task


def means_from_result(result) -> dict:
    vals = defaultdict(list)
    for ir in result.item_results:
        for ev in ir.evaluations:
            if ev.value is not None:
                vals[ev.name].append(ev.value)
    return {name: sum(v) / len(v) for name, v in vals.items()}


def run_one(name: str, cfg: dict, evaluators: list, dataset=None, data=None):
    task = make_task(cfg)
    # SDK appends its own timestamp to the run name for uniqueness; metadata records
    # what actually defines the run (config + which reranker, if any).
    meta = {**cfg, "reranker": RERANK_MODEL if cfg["rerank"] else None}
    if dataset is not None:
        result = dataset.run_experiment(name=f"ragu-{name}", metadata=meta, task=task, evaluators=evaluators)
    else:
        result = langfuse.run_experiment(name=f"ragu-{name}", data=data, metadata=meta, task=task, evaluators=evaluators)

    means = means_from_result(result)
    print(f"\n[{name}] mean scores  ({result.dataset_run_url or 'local run'}):")
    for metric, value in means.items():
        print(f"  {metric:24} {value:.3f}")
    return means


def smoke_data(dataset) -> list:
    items = list(dataset.items)
    # 2 gold-id + 1 constraint + 1 unanswerable exercise every evaluator.
    with_gold = [it for it in items if (it.expected_output or {}).get("reference_context_ids")]
    constraint = [it for it in items if (it.expected_output or {}).get("constraints")]
    unanswerable = [it for it in items if (it.metadata or {}).get("bucket") == "unanswerable"]
    picked = with_gold[:2] + constraint[:1] + unanswerable[:1]
    return [
        {"input": it.input, "expected_output": it.expected_output, "metadata": it.metadata}
        for it in picked
    ]


def run_experiment(smoke: bool, sweep: bool, config: str | None = None):
    dataset = langfuse.get_dataset(DATASET_NAME)
    src = {"data": smoke_data(dataset)} if smoke else {"dataset": dataset}
    suffix = "-smoke" if smoke else ""

    if config:
        run_one(f"sweep-{config}{suffix}", SWEEP_CONFIGS[config], RETRIEVAL_EVALUATORS, **src)
    elif sweep:
        means = {}
        for name, cfg in SWEEP_CONFIGS.items():
            means[name] = run_one(f"sweep-{name}{suffix}", cfg, RETRIEVAL_EVALUATORS, **src)
        print("\n=== sweep comparison ===")
        print(pd.DataFrame(means).T.to_string())
    else:
        run_one(
            f"baseline{suffix}",
            DEFAULT_CONFIG,
            RETRIEVAL_EVALUATORS + GENERATION_EVALUATORS,
            **src,
        )

    langfuse.flush()  # short-lived script: flush before exit or scores are lost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on 4 examples only")
    parser.add_argument("--sweep", action="store_true", help="hybrid x rerank config sweep, retrieval metrics only")
    parser.add_argument("--config", choices=list(SWEEP_CONFIGS), help="run a single sweep config (retrieval metrics only)")
    args = parser.parse_args()
    run_experiment(smoke=args.smoke, sweep=args.sweep, config=args.config)


if __name__ == "__main__":
    main()
