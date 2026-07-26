"""Review retrieval: semantic search over reviews, prefiltered by RecipeId.

Backs the agent's second tool. `search_recipes` returns recipe ids; those ids feed
`retrieve_reviews` as a hard filter, then dense similarity ranks the surviving
reviews by relevance to the query. Dense-only collection (no BM25) — see
notebook 12.
"""

from langfuse import observe
from qdrant_client.models import Filter, FieldCondition, MatchAny

from ricettario.core._shared import get_embedding, qdrant_client

REVIEWS_COLLECTION = "Recipes-reviews-collection-01"


def payload_to_review(payload, score=None):
    return {
        "recipe_id": int(payload["RecipeId"]),  # int so it ties back to search_recipes ids
        "review": payload["Review"],
        "rating": payload.get("Rating"),
        "score": score,
    }


@observe(as_type="retriever", name="retrieve_reviews")
def retrieve_reviews(query, recipe_ids, k=5, collection_name=REVIEWS_COLLECTION):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using="text-embedding-3-small",
        query_filter=Filter(must=[FieldCondition(key="RecipeId", match=MatchAny(any=recipe_ids))]),
        limit=k,
    )
    return [payload_to_review(p.payload, p.score) for p in results.points]
