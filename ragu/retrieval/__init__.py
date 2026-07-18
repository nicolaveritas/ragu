"""Retrieval package: recipe search + shared primitives.

Re-exports the public surface so callers write `from ragu.retrieval import
retrieve_data` and never reach into submodules. Add reviews here the same way
once ragu/retrieval/reviews.py exists.
"""

from ragu.retrieval._shared import PROMPTS_DIR, RERANK_MODEL, get_embedding, rerank
from ragu.retrieval.recipes import (
    RECIPES_COLLECTION,
    fetch_recipes_by_ids,
    format_blocks,
    retrieve_data,
)

__all__ = [
    "PROMPTS_DIR",
    "RERANK_MODEL",
    "RECIPES_COLLECTION",
    "get_embedding",
    "rerank",
    "retrieve_data",
    "fetch_recipes_by_ids",
    "format_blocks",
]
