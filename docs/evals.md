# Cheatsheet — Evals RAG

Vocabulary of the concepts used in this project's evals, with concrete
references to how we use them (dataset `ragu-evaluation-dataset`, script
`evals/eval_retriever.py`, notebook `06-rag-eval-dataset.ipynb`).

## LangSmith objects

| Term | What it is |
|---|---|
| **Dataset** | The collection of test questions with their ground truth. Ours: 60 LLM questions + 12 constraint. |
| **Example** | A row of the dataset: `inputs` (the question), `outputs` (the ground truth), `metadata` (labels for filtering: `bucket`, `query_style`). |
| **Experiment** | A complete run of the pipeline on the whole dataset + the scores. One experiment = one version of the pipeline (dense, hybrid, hybrid+filters...). |
| **Run / trace** | The execution of the pipeline on a single example: query, retrieved recipes, prompt, answer. You open it by clicking the row. |
| **Feedback** | The scores attached to a run, one per evaluator. The columns of the table. |
| **Evaluator** | Function `(run, example) -> score`. It receives what the pipeline produced and the ground truth, and returns a number — or `None` to say "I don't apply to this row". |
| **Compare view** | You select 2+ experiments on the same dataset and see row by row where the scores change. It's the tool for answering "did hybrid search improve things?". |

## Structure of our dataset

| Term | What it is |
|---|---|
| **Ground truth / golden (set)** | The expected correct answer. For retrieval: the ids of the correct recipes (`reference_context_ids`). "Golden" and "ground truth" are synonyms. |
| **Bucket** | Category of the question, in `metadata.bucket`: `single` (1 correct recipe), `multi` (multiple recipes), `unanswerable` (none: the system must say "I have nothing"), `constraint` (numeric constraint, gold computed from pandas). |
| **query_style** | How the question is phrased: `keyword` ("vegan cookies no eggs"), `natural` (full sentence), `detailed` (long multi-attribute request). It's used to understand on which styles a retriever performs better (BM25 should help with keywords). |
| **Constraint** | Machine-readable numeric constraint in the outputs, e.g. `{"field": "Calories", "op": "lt", "value": 300}`. It is the ground truth itself for the constraint bucket. |
| **constraint_matching_ids** | The ids that satisfy the constraint, precomputed by pandas. Saved only for inspection: no metric reads them, because verifying the constraint on the payload is equivalent to verifying membership in this list. |

## Retrieval metrics (deterministic, without LLM)

They evaluate the `k=5` retrieved recipes, before the generator writes the answer.
They apply only to rows with golden ids (they skip `unanswerable` and `constraint`).

| Metric | Formula | Answers |
|---|---|---|
| **context_recall** | golden found / total golden | "Of the correct recipes, how many are in the top-5?" |
| **hit@k (hit rate)** | 1 if at least one golden is in the top-5, otherwise 0 | "Is there at least one correct answer?" |
| **MRR** (Mean Reciprocal Rank) | 1/position of the first golden (1st → 1.0, 3rd → 0.33, absent → 0) | "Is the correct answer near the top or the bottom?" |
| **constraint_satisfaction** | retrieved recipes that respect the constraint / k | "Does everything you proposed respect the numeric constraint?" It's equivalent to a precision@k against the implicit golden. Only bucket `constraint`. |

Why there's no classic **precision** (`golden found / k`): with only 1 golden
recipe and k=5 the theoretical maximum is 0.2, so the aggregate number mixes
non-comparable buckets. hit@k and MRR measure the same thing in a readable way.

## Generation metrics (LLM-as-judge)

They evaluate the **final generated answer**, not the retrieval. An LLM ("judge",
for us gpt-5.4-mini via RAGAS) reads the answer/context and assigns the score:
useful for properties that can't be computed deterministically, but noisy — always
verify the judge's reasoning in the trace on a sample basis.

| Metric | Answers |
|---|---|
| **faithfulness** | "Are the claims in the answer supported by the retrieved recipes, or did the model make things up?" (measures hallucinations) |
| **response_relevancy** | "Does the answer address the question asked?" It judges the text of the answer: a wrong retrieval lowers it only indirectly (if the final answer ends up off-topic). |

## Retrieval concepts

| Term | What it is |
|---|---|
| **Dense embedding / semantic search** | Text → dense vector (OpenAI text-embedding-3-small); closeness in the vector ≈ closeness in meaning. Finds "porridge" for "light breakfast". Our current only retriever. |
| **Sparse embedding / BM25 / keyword search** | Lexical match weighted by term rarity. Finds the exact, rare terms that dense "blurs" (ingredient names, acronyms). No notion of meaning nor of `<`/`>`. |
| **Hybrid search** | Dense + sparse together, results merged (typically with RRF, reciprocal rank fusion). Covers both meaning and exact terms. |
| **Payload / metadata filter** | Exact filter on Qdrant's structured columns (`Calories < 500`). A hard guarantee that no retriever can give. Composable with any vector search. |
| **Re-ranking** | Second pass: a more expensive model reorders the retrieved top-N to put them in a better order. Improves MRR/precision, not recall (it doesn't add candidates). |
| **k / top-k** | How many results you ask the retriever for (5 for us). Raising k helps recall and dilutes precision. |
| **Eval sample collection** | `Recipes-collection-01-eval-sample-100`: the 100 recipes of the sample in a dedicated collection, so that every golden is guaranteed to be in the index (a missing golden would look like a retriever error: false negative). |

## How to read an experiment

1. Dataset → **Experiments** tab → click the experiment: one row per question, one column per metric (averages at the top). Empty cell = evaluator skipped for that row, not an error.
2. **Aggregate averages are misleading**: filter/group by `metadata.bucket` and `query_style` before drawing conclusions (e.g. low relevancy might come entirely from the unanswerable bucket).
3. **Filter by low score** (e.g. `hit_at_k = 0`) and open the traces: that's where you understand *why* it got it wrong.
4. To compare pipeline versions: same dataset, one experiment per version, **Compare**. Never compare experiments made on regenerated datasets (the example ids change).
