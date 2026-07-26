"""ricettario: recipe and review knowledge, served two ways from one process.

MCP at /mcp for the agent, HTTP at /api/v1 for the app. One process so the embedder,
the cross-encoder reranker and the Qdrant client load once.

Run: uv run uvicorn ricettario.app:app --reload --port 8001
"""

from fastapi import FastAPI

from ricettario.adapters import http
from ricettario.adapters.mcp import mcp

# path="/" because we mount at /mcp; the lifespan must reach FastAPI or the MCP
# session manager never initialises.
mcp_app = mcp.http_app(path="/")

app = FastAPI(title="ricettario", lifespan=mcp_app.lifespan)
app.include_router(http.router, prefix="/api/v1")
app.mount("/mcp", mcp_app)
