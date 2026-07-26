"""HTTP adapter: recipe cards by id, for the app.

Not an MCP tool: the UI hydrates its cards after the agent has answered, so it must
not depend on the model choosing to call anything.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from ricettario.core import fetch_recipes_by_ids

router = APIRouter()


@router.get("/recipes")
def get_recipes(ids: Annotated[list[int], Query(description="Recipe ids, in display order")] = []):
    return fetch_recipes_by_ids(ids)
