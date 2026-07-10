# Evals — RAGù

What we measure when we score the recipe RAG pipeline, and what each number
actually means. Two levels per metric: **In plain words** (anyone can read it)
and **Under the hood** (the exact definition + gotchas).

Everything here is scored against the dataset **`ragu-evaluation-dataset-large`**
(33 questions; source of truth `data/eval_dataset_large.jsonl`, built in
`notebooks/10-rag-eval-dataset-large.ipynb`) by `evals/eval_retriever.py`, and
tracked in **Langfuse**. Glossary of the moving parts is at the bottom.

---

## The metrics we track

Three families: **retrieval** (cheap, exact, no LLM), **generation** (an LLM
judges the written answer), and one **refusal** check.

### Retrieval metrics — deterministic, no LLM

They score the top-`k` retrieved recipes *before* the answer is written. No LLM,
so they're cheap, exact and repeatable — these run in **every** experiment
(baseline *and* the retriever sweep).

#### `context_recall`
**In plain words:** Of all the recipes that *should* have come up for this
question, what fraction did the search actually find?

**Under the hood:** `|gold ∩ retrieved| / |gold|` — the labelled correct ids
(`reference_context_ids`) intersected with the top-`k` retrieved ids, over the
number of correct ids. It's *recall*: it punishes misses but ignores ordering
and ignores irrelevant extras in the top-k. `1.0` = every correct recipe made it
into the top-5. Runs only on questions that *have* gold ids (`single`/`multi`);
skipped for `unanswerable` and `constraint`.

#### `hit@k`
**In plain words:** Did the search surface *at least one* correct recipe in the
top few results?

**Under the hood:** `1.0` if any top-`k` id is in the gold set, else `0.0`. A
lenient floor — one hit is enough, no matter how many golds exist or where they
rank. It's the metric that matters when the user only needs one good answer.
Same applicability as `context_recall` (`single`/`multi`).

#### `mrr` (Mean Reciprocal Rank)
**In plain words:** How high up the list did the *first* correct recipe appear?

**Under the hood:** `1 / rank` of the first gold id in the retrieved list (rank 1
→ `1.0`, rank 3 → `0.33`, none → `0.0`). Unlike recall/hit@k it *is* sensitive to
ordering — so this is the number re-ranking is supposed to move. `single`/`multi`
only.

> We deliberately don't track classic **precision@k** (`gold found / k`): with
> one gold recipe and `k=5` the ceiling is `0.2`, so the average just blends
> non-comparable buckets. `hit@k` and `mrr` say the same thing more readably.

#### `constraint_satisfaction`
**In plain words:** Of the recipes the system suggested, how many actually obey
the numeric limit the user asked for (e.g. "under 300 calories")?

**Under the hood:** fraction of the retrieved recipes whose Qdrant payload
satisfies *every* constraint (e.g. `{field: Calories, op: lt, value: 300}`). A
missing payload value counts as a violation (can't verify it → fail closed);
empty retrieval scores `0`. Effectively precision@k against the implicit "meets
the constraint" gold. Runs only on the `constraint` bucket. (We check the payload
directly rather than the precomputed `constraint_matching_ids` — it's equivalent
and doesn't depend on the id label being complete.)

### Generation metrics — LLM-as-judge (RAGAS)

They score the **final written answer**, not the retrieval. An LLM judge
(`gpt-5.4-mini` via RAGAS) reads the answer/context and assigns the score:
powerful for things you can't compute deterministically, but **noisy** —
spot-check the judge's reasoning in the trace. These run **only in the baseline**
(the sweep changes the retriever, not the generator, so re-judging every config
would just re-measure the same generator).

#### `faithfulness`
**In plain words:** Is everything the answer claims actually backed by the
retrieved recipes, or did the model make things up?

**Under the hood:** RAGAS breaks the answer into atomic claims and checks each
against the retrieved context; the score is the supported fraction (a
hallucination measure). **Skipped for `unanswerable`**: an honest refusal ("no
match; closest are…") decomposes into unsupported claims and scores near `0` even
when it's exactly the right behaviour. Runs on `single`/`multi`/`constraint`.

#### `response_relevancy`
**In plain words:** Does the answer actually address the question that was asked?

**Under the hood:** RAGAS generates questions *from* the answer and measures their
similarity to the real question (embedding cosine) — high when the answer is
on-topic, low when it rambles or dodges. It judges the answer text, so bad
retrieval only lowers it indirectly (if the answer ends up off-topic). Runs on
every bucket.

### Refusal metric — LLM-as-judge

#### `correctly_declined`
**In plain words:** When no recipe could possibly match, did the system honestly
say so instead of pretending?

**Under the hood:** a binary judgment from our own `gpt-5.4-mini` + the
`decline_judge` prompt — `1.0` if the answer clearly states that no available
recipe matches, else `0.0`. Offering *labelled* alternatives is fine; presenting
a recipe as if it fits = `0`. It's the mirror of faithfulness's skip: the metric
that owns the `unanswerable` bucket. The judge's reasoning is saved to the score
comment. Runs only on `unanswerable`.

---

## Which metrics run on which question

Each metric applies only to the buckets where it makes sense; elsewhere the
evaluator returns nothing and Langfuse shows an **empty cell** (skipped, *not* a
failure or a zero).

| Bucket (n) | recall / hit@k / mrr | constraint_satisfaction | faithfulness | response_relevancy | correctly_declined |
|---|:---:|:---:|:---:|:---:|:---:|
| `single` (12)      | ✓ | – | ✓ | ✓ | – |
| `multi` (4)        | ✓ | – | ✓ | ✓ | – |
| `constraint` (12)  | – | ✓ | ✓ | ✓ | – |
| `unanswerable` (5) | – | – | – | ✓ | ✓ |

---

## Glossary & concepts (reference)

### Langfuse objects

| Term | What it is |
|---|---|
| **Dataset** | The collection of test questions + their ground truth. Ours: `ragu-evaluation-dataset-large`, 33 items (21 LLM-generated + 12 constraint). Source of truth is `data/eval_dataset_large.jsonl`; `evals/upload_dataset.py` pushes it to Langfuse (idempotent upsert by question hash). |
| **Dataset item** | One row: `input` (the question), `expected_output` (the ground truth), `metadata` (labels for filtering: `bucket`, `query_style`). |
| **Experiment** (dataset run) | One full run of the pipeline over the dataset + its scores. One experiment = one pipeline config (dense, hybrid, hybrid+rerank…). Created by `dataset.run_experiment(name, task, evaluators)`. |
| **Trace** | The execution on a single item: query → retrieved recipes → prompt → answer. Click a row to open it. |
| **Score** | A metric value attached to a trace/run — one per evaluator. The columns of the results table. |
| **Evaluator** | Function `(*, input, output, expected_output, metadata) -> Evaluation(name, value, comment)`. Returns `[]` to say "I don't apply to this row" (records no score). |
| **Compare** | Select 2+ experiments on the *same* dataset and see per-row score deltas — the tool for "did hybrid search improve things?". |

### Our dataset structure

| Term | What it is |
|---|---|
| **Ground truth / golden** | The expected correct answer. For retrieval: the ids of the correct recipes (`reference_context_ids`). Only `single`/`multi` questions have these. "Golden" and "ground truth" are synonyms. |
| **Bucket** | Category of the question (`metadata.bucket`): `single` (1 correct recipe), `multi` (several), `unanswerable` (none — the system must decline), `constraint` (numeric constraint, gold computed with pandas). |
| **query_style** | How the question is phrased (`metadata.query_style`): `keyword` ("vegan cookies no eggs"), `natural` (full sentence), `detailed` (long multi-attribute request). Used to see which styles a retriever handles better (BM25 should help keywords). |
| **Constraint** | Machine-readable numeric constraint in `expected_output`, e.g. `{"field": "Calories", "op": "lt", "value": 300}`. It *is* the ground truth for the `constraint` bucket. |
| **constraint_matching_ids** | The ids that satisfy the constraint, precomputed by pandas. Kept for inspection only — no metric reads them (checking the payload directly is equivalent). |

### Retrieval concepts

| Term | What it is |
|---|---|
| **Dense embedding / semantic search** | Text → dense vector (OpenAI `text-embedding-3-small`); closeness in vector space ≈ closeness in meaning. Finds "porridge" for "light breakfast". |
| **Sparse embedding / BM25 / keyword search** | Lexical match weighted by term rarity. Finds exact, rare terms that dense "blurs" (ingredient names, acronyms). No notion of meaning nor of `<`/`>`. |
| **Hybrid search** | Dense + sparse together, results merged (typically RRF, reciprocal rank fusion). Covers both meaning and exact terms. |
| **Payload / metadata filter** | Exact filter on Qdrant's structured fields (`Calories < 500`). A hard guarantee no vector search can give. Composable with any retriever. |
| **Re-ranking** | Second pass: a more expensive model reorders the retrieved top-N. Improves `mrr`/precision, *not* recall (it adds no candidates). |
| **k / top-k** | How many results you ask the retriever for (5 for us). Higher `k` helps recall, dilutes precision. |
| **Eval collection** | We eval against the **full** collection `Recipes-collection-01-hybrid`. The gold labels are complete over the whole corpus (relevance-swept + verified), so no dedicated sample collection is needed. |

### How to read an experiment

1. Dataset → **Runs** → open a run: one row per question, one column per metric (averages up top). **Empty cell = evaluator skipped for that row**, not an error or a zero.
2. **Aggregate averages are misleading**: group by `metadata.bucket` / `query_style` before concluding (e.g. a low `response_relevancy` might come entirely from the `unanswerable` bucket).
3. **Filter by low score** (e.g. `hit@k = 0`) and open the traces — that's where you see *why* it got it wrong.
4. To compare pipeline versions: same dataset, one experiment per version, **Compare**. Never compare experiments run on regenerated datasets (item ids change).
