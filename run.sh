#!/usr/bin/env bash
# Dev launcher: full stack (Qdrant + Langfuse) in Docker, ricettario + API + UI locally
# with hot reload. Ctrl-C stops the local apps; `docker compose down` stops the containers.
set -euo pipefail

docker compose up -d

trap 'kill $knowledge $api $web 2>/dev/null' EXIT
# ricettario first: the agent loads its tools from /mcp on the first question.
uv run uvicorn ricettario.app:app --reload --port 8001 & knowledge=$!
uv run uvicorn api:app --reload --port 8000 & api=$!
uv run streamlit run app.py & web=$!
wait
