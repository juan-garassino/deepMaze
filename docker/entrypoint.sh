#!/usr/bin/env bash
# Backend entrypoint. Runs FastAPI via uvicorn.
# Env: CORS_ORIGINS (comma-separated), PORT (default 8000).
set -euo pipefail

cd /app

# Make sibling package dirs importable.
for sub in agents environment training utils config web; do
  export PYTHONPATH="/app/$sub:${PYTHONPATH:-}"
done

exec uvicorn web.server:create_app \
  --factory \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --log-level info
