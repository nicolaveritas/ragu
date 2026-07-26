"""FastAPI service exposing the ragu agent. Run: uvicorn api:app --reload

Recipe cards are hydrated from ricettario over HTTP once the agent has answered.
"""

import os

import httpx
from fastapi import FastAPI
from langfuse import get_client
from pydantic import BaseModel

from ragu.agent import run_agent

RICETTARIO_URL = os.getenv("RICETTARIO_URL", "http://localhost:8001")

app = FastAPI(title="Ragù API")
langfuse = get_client()
ricettario = httpx.AsyncClient(base_url=RICETTARIO_URL, timeout=30)


class Query(BaseModel):
    question: str
    thread_id: str


class Feedback(BaseModel):
    trace_id: str
    value: bool  # thumbs up = True, thumbs down = False
    comment: str = ""


@app.post("/chat")
async def chat(q: Query):
    result = await run_agent(q.question, q.thread_id)
    response = await ricettario.get(
        "/api/v1/recipes", params={"ids": [r["id"] for r in result["references"]]}
    )
    response.raise_for_status()
    return {
        "question": result["question"],
        "answer": result["answer"],
        "recipes": response.json(),
        "trace_id": result["trace_id"],
    }


@app.post("/feedback")
def feedback(f: Feedback):
    langfuse.create_score(
        trace_id=f.trace_id,
        name="user-feedback",
        value=int(f.value),  # BOOLEAN scores take 0/1
        data_type="BOOLEAN",
        comment=f.comment or None,
    )
    return {"ok": True}
