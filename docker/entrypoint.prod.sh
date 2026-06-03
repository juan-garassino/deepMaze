#!/usr/bin/env bash
# Background the GCS asset sync so it doesn't block the startup probe.
# Models become available a few seconds after the service is healthy.
set -euo pipefail
cd /app

if [[ -n "${ASSETS_BUCKET:-}" ]]; then
  ( python docker/sync_assets.py \
      || echo "[entrypoint] WARN: asset sync failed; baked-in assets only" ) &
fi

exec gunicorn \
  --bind "0.0.0.0:${PORT:-8000}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${WEB_CONCURRENCY:-1}" \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  "web.server:create_app()"
