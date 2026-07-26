"""MCP adapter: the two tools the agent calls, over streamable HTTP (or stdio).

The docstrings below are prompt, not documentation: FastMCP ships them to the model
as the tool descriptions, and they are all it has to decide when to call a tool and
what to put in the arguments. Editing them changes agent behaviour.
"""

from fastmcp import FastMCP

from ricettario.adapters.render import reviews_to_llm_text, to_llm_text
from ricettario.core import retrieve_and_rerank, retrieve_reviews

mcp = FastMCP("ricettario")


@mcp.tool
def search_recipes(query: str, top_k: int = 5) -> str:
    """Search the recipe database and return the most relevant recipes.

    Args:
        query: Natural-language search query. Keep any numeric constraints in
            the query text (e.g. "under 300 calories", "at least 20g protein",
            "ready in 30 minutes"): they are extracted automatically and
            applied as hard filters on the search.
        top_k: Number of recipes to return. Works best with 5 or more.

    Returns:
        One text block per recipe with name, id, rating, nutrition facts,
        total time, ingredients and steps. Starts with a note if no recipe
        satisfied the numeric constraints (closest matches are shown instead).
    """
    found = retrieve_and_rerank(query, top_k=top_k)
    return to_llm_text(found["recipes"], filter_relaxed=found["filter_relaxed"])


@mcp.tool
def search_reviews(recipe_ids: list[int], query: str, top_k: int = 5) -> str:
    """Search user reviews for specific recipes.

    Call this AFTER `search_recipes`, to find out what people say about recipes
    it returned (taste, texture, difficulty, ingredient swaps, whether it works
    as written). Reviews are searched only within the recipes you name.

    Args:
        recipe_ids: The recipe `id` values from `search_recipes` results whose
            reviews you want. Pass one or several.
        query: What to look for in the reviews (e.g. "too sweet", "gluten-free
            substitutions", "kids liked it"). Reviews are ranked by relevance to
            this.
        top_k: Number of reviews to return across all the given recipes.

    Returns:
        One line per review, tagged with its recipe id and star rating.
    """
    return reviews_to_llm_text(retrieve_reviews(query, recipe_ids=recipe_ids, k=top_k))
