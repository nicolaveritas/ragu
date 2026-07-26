"""FastAPI service exposing the ragu agent. Run: uvicorn api:app --reload

Recipe cards are hydrated from ricettario over HTTP once the agent has answered.
"""

import json
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langfuse import get_client
from pydantic import BaseModel

from ragu.agent import stream_agent

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


async def sse(question: str, thread_id: str):
    """Re-emit agent events as SSE frames: `data: {json}\\n\\n`, one JSON object per frame.

    Recipe cards are hydrated here, on the way out, so the status frames reach the
    browser while the agent works and the card fetch only delays the last frame.
    """
    async for event in stream_agent(question, thread_id):
        if event["type"] == "final":
            response = await ricettario.get(
                "/api/v1/recipes", params={"ids": [r["id"] for r in event["references"]]}
            )
            response.raise_for_status()
            event = {**event, "recipes": response.json()}
        yield f"data: {json.dumps(event)}\n\n"


@app.post("/chat")
async def chat(q: Query):
    return StreamingResponse(sse(q.question, q.thread_id), media_type="text/event-stream")


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
