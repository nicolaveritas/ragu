"""FastAPI service exposing the ragu agent. Run: uvicorn api:app --reload"""

from fastapi import FastAPI
from pydantic import BaseModel

from ragu.agent import run_agent
from ragu.retrieval import fetch_recipes_by_ids

app = FastAPI(title="Ragù API")


class Query(BaseModel):
    question: str
    thread_id: str


@app.post("/chat")
def chat(q: Query):
    result = run_agent(q.question, q.thread_id)
    recipes = fetch_recipes_by_ids([r["id"] for r in result["references"]])
    return {"question": result["question"], "answer": result["answer"], "recipes": recipes}
