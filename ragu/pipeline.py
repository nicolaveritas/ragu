"""
Legacy linear RAG pipeline: retrieve -> format -> generate, single shot.

Superseded by the LangGraph agent (agent.py). Kept only as the retriever-eval
baseline (evals/eval_retriever.py). Delete once that eval migrates to the agent.
"""

from ragu.retrieval import (  # noqa: F401  (RERANK_MODEL re-exported for the eval)
    PROMPTS_DIR,
    RECIPES_COLLECTION,
    RERANK_MODEL,
    format_blocks,
    rerank,
    retrieve_data,
)

from langfuse import observe
from langfuse.openai import openai

from ragu.prompt_loader import render_prompt


@observe(name="build_prompt")
def build_system_prompt(context, constraints_not_satisfied=False):
    return render_prompt(
        PROMPTS_DIR,
        "system_prompt",
        context=context,
        constraints_not_satisfied=constraints_not_satisfied,
    )


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
def rag_pipeline(question, k=5, candidates=20, collection_name=RECIPES_COLLECTION, hybrid=True, use_rerank=True):
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
