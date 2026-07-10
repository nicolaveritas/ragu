"""Upload the eval dataset to Langfuse.

Source of truth is data/eval_dataset_large.jsonl (built in notebooks/10), NOT
Langfuse — this script is just the thin "sink" that pushes it up. Idempotent:
each item gets a stable id derived from its question, so re-running upserts in
place instead of creating duplicates. (It does not delete items removed from the
JSONL - archive those in the UI if needed. ponytail: no bulk-delete, YAGNI.)

Usage (from repo root, with Langfuse running on localhost:3000):
    uv run python evals/upload_dataset.py
"""

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # env before langfuse import

from langfuse import get_client

DATASET_NAME = "ragu-evaluation-dataset-large"
DATA_PATH = Path(__file__).parent.parent / "data" / "eval_dataset_large.jsonl"

langfuse = get_client()


def item_id(question: str) -> str:
    return hashlib.sha1(question.encode()).hexdigest()[:16]


def main():
    records = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]

    langfuse.create_dataset(
        name=DATASET_NAME,
        description=(
            "RAG eval dataset over the full recipe corpus; built in notebooks/10. "
            "Source of truth: data/eval_dataset_large.jsonl."
        ),
    )

    for r in records:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item_id(r["question"]),
            input={"question": r["question"]},
            expected_output={
                "reference_context_ids": r["reference_context_ids"],
                "constraints": r["constraints"],
                "constraint_matching_ids": r["constraint_matching_ids"],
                "ground_truth": r["answer_example"],
                "reference_description": r["reference_description"],
            },
            metadata={"bucket": r["bucket"], "query_style": r["query_style"]},
        )

    langfuse.flush()
    print(f"upserted {len(records)} items into dataset '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
