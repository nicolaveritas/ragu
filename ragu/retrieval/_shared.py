"""Shared retrieval primitives: the Qdrant client, embedding, and reranking.

Capability-agnostic. Recipe retrieval (recipes.py) and any future retriever
(reviews, ...) build on these. No knowledge of any specific payload shape.
"""

import os
from pathlib import Path

from flashrank import Ranker, RerankRequest
from langfuse import observe
from langfuse.openai import openai  # drop-in replacement: auto-traces every OpenAI call
from qdrant_client import QdrantClient

# ragu/prompts — this file is ragu/retrieval/_shared.py, so parent.parent == ragu/
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
ranker = Ranker(model_name=RERANK_MODEL)


def get_embedding(text, model="text-embedding-3-small"):
    # langfuse.openai auto-logs an embedding generation with model + token usage;
    # `name` just labels it in the trace. No manual usage bookkeeping needed.
    response = openai.embeddings.create(input=text, model=model, name="embed_query")
    return response.data[0].embedding


@observe(as_type="retriever", name="rerank")
def rerank(query, items, top_n=5):
    """Rerank any list of dicts carrying a "text" key; sets each survivor's "score".

    Generic over payload shape — used by every retriever, not just recipes.
    """
    if not items:  # ponytail: ranker chokes on an empty/all-blank document list
        return []
    docs = [{"id": i, "text": item.get("text")} for i, item in enumerate(items)]
    response = ranker.rerank(RerankRequest(query=query, passages=docs))
    out = []
    for result in response[:top_n]:
        item = items[result["id"]]
        item["score"] = result["score"]
        out.append(item)
    return out
