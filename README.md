# 🍝 RAGù

A retrieval-augmented generation (RAG) system built on the [Food.com recipes & reviews dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) — over 500k recipes and 1.4M user reviews, with structured fields (cooking time, nutrition, tags, ratings) alongside free-text descriptions, steps, and reviews.

## Goals

The aim is to go beyond a basic proof-of-concept and explore a realistic RAG stack on this data:

- Ingestion and embedding of recipes as a searchable knowledge base
- Structured outputs and metadata-aware retrieval
- Hybrid search combining semantic similarity with structured filters (e.g. *"vegetarian recipes under 30 minutes and 500 kcal"*)
- A synthetic, partly verifiable evaluation set to measure retrieval and generation quality
- Agentic extensions (e.g. a meal-planning assistant that coordinates recipe search, nutrition balancing, and shopping lists)

## Dataset

The dataset is **not included** in this repository — it has its own usage terms and shouldn't be redistributed here. Download it from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
```

Then point the ingestion pipeline at the downloaded `recipes.parquet` / `reviews.parquet`.

## Running

Three local processes: **ricettario** owns all knowledge access (recipe/review search over
MCP for the agent, recipe cards over HTTP for the app), the **API** runs the agent, and the
Streamlit **UI** talks to the API over HTTP. Copy `.env.example` to `.env` and fill in the
keys, then:

```bash
./run.sh
```

This starts Qdrant + Langfuse (Docker) and runs ricettario (:8001, MCP at `/mcp`, HTTP at
`/api/v1`), the API (:8000, docs at `/docs`) and the UI (:8501) with hot reload; Ctrl-C stops
the local apps, `docker compose down` the containers. Set `RAGU_API_URL`, `RICETTARIO_URL`
or `RICETTARIO_MCP_URL` if anything lives elsewhere.

## Status

Work in progress.