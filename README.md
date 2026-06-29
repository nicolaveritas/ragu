# Ragù

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

## Status

Work in progress.