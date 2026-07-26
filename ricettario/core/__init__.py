"""Core knowledge access: recipe + review retrieval and the shared primitives.

Plain functions, no server needed: evals/eval_retriever.py and the notebooks import
them in-process.

Re-exports the public surface so callers write `from ricettario.core import
retrieve_data` and never reach into submodules.
"""

from ricettario.core._shared import RERANK_MODEL, get_embedding, rerank
from ricettario.core.recipes import (
    RECIPES_COLLECTION,
    fetch_recipes_by_ids,
    format_blocks,
    retrieve_and_rerank,
    retrieve_data,
)
from ricettario.core.reviews import (
    REVIEWS_COLLECTION,
    retrieve_reviews,
)

__all__ = [
    "RERANK_MODEL",
    "RECIPES_COLLECTION",
    "REVIEWS_COLLECTION",
    "get_embedding",
    "rerank",
    "retrieve_data",
    "retrieve_and_rerank",
    "fetch_recipes_by_ids",
    "format_blocks",
    "retrieve_reviews",
]
