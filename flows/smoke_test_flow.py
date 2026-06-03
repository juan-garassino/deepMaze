"""Prefect flow: daily smoke test against the deployed Cloud Run service.

POSTs to /api/inference with a known model and asserts the SSE stream
reaches a terminal episode event.

Bootstrap requirement: `assets/${SMOKE_MODEL_NAME}/` must exist in the
deployed service (either baked into the image or available in
`gs://${ASSETS_BUCKET}/`). The first promote_flow run is what gets the
smoke test out of "no such model" failure mode.
"""

from __future__ import annotations

import json
import os

import httpx
from prefect import flow, get_run_logger, task

CLOUD_RUN_URL = os.environ.get("CLOUD_RUN_URL", "")
MODEL_NAME = os.environ.get("SMOKE_MODEL_NAME", "drqn_v1")
MODEL_SOURCE = os.environ.get("SMOKE_MODEL_SOURCE", "asset")
TIMEOUT_S = 60


@task
def post_inference() -> bool:
    log = get_run_logger()
    if not CLOUD_RUN_URL:
        raise RuntimeError("CLOUD_RUN_URL not set")
    url = f"{CLOUD_RUN_URL.rstrip('/')}/api/inference"
    payload = {"source": MODEL_SOURCE, "name": MODEL_NAME,
               "maze_source": "trained"}
    with httpx.stream("POST", url, json=payload, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "episode" and ev.get("done"):
                log.info(f"episode complete: reward={ev.get('total_reward')}")
                return True
    return False


@flow(name="smoke_test_flow")
def smoke_test_flow() -> bool:
    ok = post_inference()
    if not ok:
        raise RuntimeError("inference stream did not yield a completed episode")
    return ok


if __name__ == "__main__":
    smoke_test_flow()
