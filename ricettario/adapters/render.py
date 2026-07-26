"""Rendering for the MCP tools: recipe and review dicts -> the text the agent reads.

`format_blocks` itself lives in core: ragu/pipeline.py builds its prompt with the
same format and evals/eval_retriever.py scores against it, so there is one copy.
"""

from ricettario.core import format_blocks


def to_llm_text(recipes, filter_relaxed=False):
    """One text block per recipe, prefixed with a note when the filter was relaxed."""
    if not recipes:
        return "No recipes found for this query. Try rephrasing or broadening it."
    text = "\n\n".join(format_blocks(recipes))
    if filter_relaxed:
        text = (
            "Note: no recipes matched the numeric constraints; "
            "showing the closest matches instead.\n\n" + text
        )
    return text


def reviews_to_llm_text(reviews):
    """One line per review, tagged with its recipe id and star rating."""
    if not reviews:
        return "No reviews found for these recipes."
    lines = []
    for r in reviews:
        rating = f"[{r['rating']}/5] " if r.get("rating") is not None else ""
        lines.append(f"- recipe {r['recipe_id']} {rating}{r['review']}")
    return "\n".join(lines)
