#!/usr/bin/env bash
# Dev launcher: full stack (Qdrant + Langfuse) in Docker, API + UI locally with hot
# reload. Ctrl-C stops the local app; `docker compose down` stops the containers.
set -euo pipefail

docker compose up -d

trap 'kill $api $web 2>/dev/null' EXIT
uv run uvicorn api:app --reload --port 8000 & api=$!
uv run streamlit run app.py & web=$!
wait
