# Prefect flows (D)

Three flows that glue the MLOps loop together. Each is a plain Python entrypoint that also works without Prefect (just `python flows/<name>_flow.py`).

## One-time setup

```bash
pip install "prefect>=3.0" mlflow httpx
prefect work-pool create --type process default-process    # one-time
prefect deploy --all --prefect-file flows/prefect.yaml     # registers all 3 deployments
```

Use Prefect Cloud (free tier) or a local server via `infra/prefect/docker-compose.yml`.

## Required env

| Var | Used by | Why |
|---|---|---|
| `MLFLOW_TRACKING_URI` | retrain, promote | reach the tracking server |
| `ASSETS_BUCKET` | promote (GCS mode) | where to upload promoted bundles |
| `REPO_DIR` | promote (PR mode) | path to a clean git checkout — defaults to `.` |
| `CLOUD_RUN_URL` | smoke_test | target of the daily probe |
| `SMOKE_MODEL_NAME` | smoke_test | which asset to invoke (default `drqn_v1`) |
| `SMOKE_MODEL_SOURCE` | smoke_test | `asset` or `run` (default `asset`) |
| `RETRAIN_POLL_SECONDS` | retrain | MLflow poll cadence (default 60s) |
| `RETRAIN_MAX_WAIT_MIN` | retrain | wall-clock cap (default 180min) |
| `RETRAIN_EVAL_METRIC` | retrain | which metric to compare champions on (default `eval_success_rate`) |

## Flow reference

### `retrain_flow`

**What it does.** Watches MLflow for a *new* finished run that beats the current champion (highest `eval_success_rate`), then calls `promote_flow` to ship it.

**When to trigger.** After kicking off a Colab notebook — the flow waits for the notebook to finish logging. Useful as a "babysit my training" loop.

**Inputs:** `open_pr: bool = True` — passed through to `promote_flow`.

**Side effects:** opens a PR (or pushes to GCS). Otherwise read-only.

**Run locally:**
```bash
export MLFLOW_TRACKING_URI=...
python flows/retrain_flow.py
```

**Caveat:** polls `search_runs` every 60s up to 3h. If MLflow is unreachable for a long stretch, the flow times out — by design.

### `promote_flow`

**What it does.** Given an MLflow run-id, downloads the `assets/<name>/` artifact, validates `config.json` against `REQUIRED_KEYS`, and either:
- **PR mode (default):** copies the bundle into `<REPO_DIR>/assets/`, creates branch `promote/<name>`, pushes, opens a PR with `gh`.
- **GCS mode:** `gsutil rsync` to `gs://${ASSETS_BUCKET}/<name>/`.

**When to trigger.** Manually after any Colab run, or automatically by `retrain_flow`.

**Inputs:** `run_id: str`, `open_pr: bool = True`.

**Side effects:** PR mode mutates git history; GCS mode mutates the bucket.

**Run locally:**
```bash
python flows/promote_flow.py <mlflow-run-id>
```

**Caveats:**
- Requires `gh` authenticated and `git` configured (PR mode).
- Requires `gsutil` + ambient GCP creds (GCS mode).
- Doesn't retry on transient MLflow errors — let Prefect handle retries via deployment config.

### `smoke_test_flow`

**What it does.** POSTs to `${CLOUD_RUN_URL}/api/inference` with a known model name and consumes the SSE stream until it sees a `done: true` episode event. Raises if no such event arrives within 60 s.

**When to trigger.** Daily at 09:00 UTC (see `flows/prefect.yaml`). Also useful as a post-deploy probe.

**Inputs:** none (all config via env).

**Side effects:** none — read-only HTTP.

**Run locally:**
```bash
export CLOUD_RUN_URL=https://deepmaze-backend-...run.app
python flows/smoke_test_flow.py
```

**Bootstrap:** fails on first-ever run because no model has been promoted yet. Run `promote_flow` at least once before enabling the schedule.

## Deployment manifest

See [`prefect.yaml`](prefect.yaml) — three deployments, only `smoke-test` is scheduled by default.

## How `flows/` resolves on a Prefect worker

A worker (local-process or Docker-based) needs to import `from flows.promote_flow import promote_flow`. Two ways that works:

1. Run the worker from the repo root — `flows/` is then directly importable.
2. Set `PYTHONPATH=$PWD` in the Prefect deployment's `job_variables`.

If you containerize the worker later, add a `Dockerfile.prefect` that `COPY . /app && WORKDIR /app` and set `PYTHONPATH=/app`.
