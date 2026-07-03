"""FastAPI service exposing the ragu RAG pipeline. Run: uvicorn api:app --reload"""

from fastapi import FastAPI
from pydantic import BaseModel

from ragu.pipeline import rag_pipeline

app = FastAPI(title="Ragù API")


class Query(BaseModel):
    question: str
    k: int = 5


@app.post("/chat")
def chat(q: Query):
    # ponytail: return the pipeline dict as-is; add a response_model when the contract needs freezing
    return rag_pipeline(q.question, k=q.k)
