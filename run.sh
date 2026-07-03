#!/usr/bin/env bash
# Dev launcher: Qdrant in Docker, API + UI locally with hot reload. Ctrl-C stops the app.
set -euo pipefail

docker compose up -d qdrant

trap 'kill $api $web 2>/dev/null' EXIT
uv run uvicorn api:app --reload --port 8000 & api=$!
uv run streamlit run app.py & web=$!
wait
