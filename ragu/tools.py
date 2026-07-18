from langchain_core.tools import tool
from ragu.retrieval import format_blocks, rerank, retrieve_data

@tool
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
    retrieved = retrieve_data(query, k=20)
    recipes = rerank(query, retrieved["recipes"], top_n=top_k)
    if not recipes:
        return "No recipes found for this query. Try rephrasing or broadening it."
    text = "\n\n".join(format_blocks(recipes))
    if retrieved["filter_relaxed"]:
        text = (
            "Note: no recipes matched the numeric constraints; "
            "showing the closest matches instead.\n\n" + text
        )
    return text