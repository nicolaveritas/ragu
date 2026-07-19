"""FastAPI service exposing the ragu agent. Run: uvicorn api:app --reload"""

from fastapi import FastAPI
from langfuse import get_client
from pydantic import BaseModel

from ragu.agent import run_agent
from ragu.retrieval import fetch_recipes_by_ids

app = FastAPI(title="Ragù API")
langfuse = get_client()


class Query(BaseModel):
    question: str
    thread_id: str


class Feedback(BaseModel):
    trace_id: str
    value: bool  # thumbs up = True, thumbs down = False
    comment: str = ""


@app.post("/chat")
def chat(q: Query):
    result = run_agent(q.question, q.thread_id)
    recipes = fetch_recipes_by_ids([r["id"] for r in result["references"]])
    return {
        "question": result["question"],
        "answer": result["answer"],
        "recipes": recipes,
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
