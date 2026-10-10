#!/usr/bin/env bash
# MemeCoin AI Trader — one-click launcher (Linux).
# Starts the local server and opens the dashboard in your browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run: creating virtual environment..."
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r requirements.txt
  echo "Done. Tip: copy .env.example to .env and add your MISTRAL_API_KEY."
fi

if [ ! -f .env ]; then
  echo "Note: no .env found. The AI cannot run decision cycles without MISTRAL_API_KEY."
  echo "      cp .env.example .env  then edit it."
fi

PORT="${MEMICOIN_PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

( sleep 1.5 && (xdg-open "$URL" >/dev/null 2>&1 || true) ) &

exec ./.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
