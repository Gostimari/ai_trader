#!/usr/bin/env bash
# MemeCoin AI Trader — one-click launcher (Linux).
# Starts the local server and opens the dashboard in your browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run: creating virtual environment..."
  python3 -m venv .venv
fi

# Sync dependencies on every start: fast no-op when already installed,
# and crucially picks up newly added requirements.txt entries.
echo "Checking dependencies..."
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
  echo "Note: no .env found. You can set the Mistral key in the app (Settings) instead."
fi

PORT="${MEMICOIN_PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

( sleep 1.5 && (xdg-open "$URL" >/dev/null 2>&1 || true) ) &

exec ./.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
