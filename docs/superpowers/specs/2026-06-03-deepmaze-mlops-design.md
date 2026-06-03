# deepMaze MLOps + GCP — meta-spec

**Date:** 2026-06-03
**Status:** in progress
**Scope:** integration contracts across six subsystems. Each subsystem will get its own dedicated spec when drilled into.

## Goal

End-to-end MLOps loop around the existing deepMaze RL playground:
1. Train heavy agents (DRQN/DTQN) externally (Colab → eventually Vertex AI).
2. Track every run in MLflow.
3. Ship the inference service to GCP (Cloud Run).
4. Wire CI/CD with Slack notifications.
5. Instrument everything so it doubles as a learning artifact.

## Repo layout additions

```
deepMaze/
├── notebooks/
│   └── train_agent.ipynb              # A — Colab notebook
├── infra/
│   ├── mlflow/                         # B — Cloud Run MLflow server
│   │   ├── Dockerfile
│   │   ├── deploy.sh
│   │   ├── docker-compose.local.yml    # dev MLflow + Postgres
│   │   └── README.md
│   ├── cloudrun/                       # C — inference Cloud Run service
│   │   └── service.yaml
│   └── prefect/                        # D — local Prefect server
│       └── docker-compose.yml
├── flows/                              # D — Prefect flows
│   ├── retrain_flow.py
│   ├── promote_flow.py
│   └── smoke_test_flow.py
├── Dockerfile.prod                     # C — slim prod backend
├── .github/workflows/
│   ├── test.yml                        # exists
│   └── deploy.yml                      # E — new GCP+Slack pipeline
└── docs/superpowers/specs/             # this folder
```

## Subsystem contracts

### A. Training notebook → artifact
- **Input:** repo at HEAD, hyperparameter overrides (form fields in Colab), `MLFLOW_TRACKING_URI`, `GOOGLE_APPLICATION_CREDENTIALS`.
- **Process:** clones repo, installs as editable, instantiates env + agent via existing factories, trains, evaluates.
- **Output:**
  - MLflow run with params, per-episode metrics, final eval, replay GIF, model checkpoint.
  - `assets/<run_name>/` bundle (`config.json` + `model.pt` + `viz/replay.webp`) — matches existing pretrained-inference contract.
- **Contract:** `config.json` schema is whatever `web/server.py::_load_pretrained` already parses. Don't change that schema; conform to it.

### B. MLflow tracking server
- **Topology:** Cloud Run service `mlflow-server` → Cloud SQL Postgres (backend store) → GCS bucket (artifact store).
- **Auth:** Cloud Run service account has `roles/cloudsql.client` + `roles/storage.objectAdmin` on the artifact bucket. Notebook + backend reach the server via signed URL or public unauthenticated (dev-only).
- **Env contract:**
  - Server reads `BACKEND_STORE_URI`, `ARTIFACT_ROOT`, `PORT`.
  - Clients set `MLFLOW_TRACKING_URI=https://mlflow-server-...run.app` + `GOOGLE_APPLICATION_CREDENTIALS` (for GCS artifact upload).
- **Local dev:** `infra/mlflow/docker-compose.local.yml` runs `mlflow` + `postgres` + a local filesystem artifact root.

### C. Inference container → GCP
- **Image:** `Dockerfile.prod` — strips dev deps, uses `gunicorn -k uvicorn.workers.UvicornWorker` for cold-start determinism.
- **Cloud Run config:** `infra/cloudrun/service.yaml` — min-instances 0, max 2, 512Mi, port 8000.
- **Asset strategy:** at startup the backend syncs `gs://${ASSETS_BUCKET}/` → `/app/assets/` (read-only). Promotion path D writes to that bucket.
- **Frontend:** Firebase Hosting (static), `window.API_BASE_URL` set at build time from the Cloud Run URL. (Decided: Firebase over Cloud Run for the SPA — cheaper, faster cold start, no envsubst hack.)

### D. Prefect orchestration
- **Server:** Prefect Cloud (free tier) — keep API key in `PREFECT_API_KEY`. Local dev compose is optional.
- **Flows:**
  - `retrain_flow(agent_type, hyperparams)` — kicks a Colab job via REST → polls MLflow for completion → fetches best run → builds `assets/<name>/` → commits + opens PR via `gh`.
  - `promote_flow(run_id)` — given an MLflow run-id, downloads artifacts, validates the bundle against the schema, copies into `gs://${ASSETS_BUCKET}/<name>/`, triggers Cloud Run redeploy.
  - `smoke_test_flow()` — daily schedule, hits `/api/inference` with a known model, asserts the returned episode reaches a terminal cell.
- **Deployments:** `prefect deploy --all` from `flows/prefect.yaml`.

### E. CI/CD with Slack
- **`test.yml` (exists):** pytest + ruff, no changes.
- **`deploy.yml` (new):** on push to `main` AND `test.yml` green:
  1. Slack notify (`start`) via `rtCamp/action-slack-notify`.
  2. `google-github-actions/auth` (OIDC, workload identity federation).
  3. `google-github-actions/setup-gcloud`.
  4. Configure docker for Artifact Registry.
  5. `docker build -f Dockerfile.prod -t ${IMAGE}:${SHA} .`
  6. `docker push ${IMAGE}:${SHA}`
  7. `gcloud run deploy ${CLOUD_RUN_SERVICE} --image ${IMAGE}:${SHA} ...`
  8. Slack notify (`success` / `failure`).
- **Secrets** (GH repo secrets): `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `GAR_REPO`, `CLOUD_RUN_SERVICE`, `SLACK_WEBHOOK_URL`.

### F. Observability
- **MLflow** — already covers experiment history + model registry.
- **Prefect** — flow run history + retries.
- **Cloud Run** — built-in revision metrics → Cloud Monitoring dashboard.
- **OTEL** — `opentelemetry-instrumentation-fastapi` in `web/server.py`, exporter → Cloud Trace via `opentelemetry-exporter-gcp-trace`. Sample rate configurable via `OTEL_TRACES_SAMPLER_ARG`.
- **(optional)** Grafana on Cloud Run reading Cloud Monitoring + MLflow Postgres — deferred.

## Integration contract table

| Producer | Artifact | Consumer |
|---|---|---|
| Notebook (A) | MLflow run + `assets/<name>/` bundle | Backend (C), Prefect (D) |
| MLflow (B) | tracking URI + model registry | A, D |
| Prefect (D) | promoted `assets/<name>/` in GCS + PR | GHA deploy (E) |
| GHA deploy (E) | Cloud Run revision | users + smoke flow (D) |
| Backend (C) | OTEL spans + structured logs | Cloud Trace / Logging (F) |

## Env vars (full table)

| Var | Where set | Who reads |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | local dev only | notebook, backend (local), MLflow client |
| `MLFLOW_TRACKING_URI` | GH secret, Colab form, backend env | notebook, backend, Prefect |
| `MLFLOW_S3_ENDPOINT_URL` | unused (we use GCS native, not S3-compat) | — |
| `ASSETS_BUCKET` | Cloud Run env, Prefect env | backend startup sync, promote_flow |
| `SLACK_WEBHOOK_URL` | GH secret | deploy.yml |
| `PREFECT_API_KEY` | local dev, Prefect-managed | flows |
| `GAR_REPO` | GH secret | deploy.yml |
| `CLOUD_RUN_SERVICE` | GH secret | deploy.yml |
| `GCP_PROJECT_ID` | GH secret + Cloud Run env | deploy.yml, backend |
| `WIF_PROVIDER` / `WIF_SERVICE_ACCOUNT` | GH secret | deploy.yml |

## Out of scope (this phase)

- Vertex AI training (Colab first).
- Multi-region Cloud Run.
- Authn on the deployed `/api/inference` (public for demo).
- Cost dashboards / budget alerts.
- Per-treasure reward shaping (separate concern; lives in deepMaze proper).

## Success criteria

1. Colab DRQN run appears in MLflow with metrics + replay artifact in one execution.
2. `git push main` → GHA → Cloud Run deploy → Slack start + end messages.
3. Prefect `promote_flow` moves any MLflow-registered model into `assets/`, opens a PR, merged PR redeploys.
4. Deployed `/api/inference` serves a greedy episode using a Colab-trained DRQN.
5. MLflow, Prefect, Cloud Run, GHA all reachable from one README-linked landing page.

## Build order

**B → A → C → E → D → F**, then docs sync.

## Docs to update

- `CLAUDE.md` — add: new top-level dirs (`infra/`, `flows/`, `notebooks/`); new env vars; new commands (`infra/mlflow/deploy.sh`, `prefect deploy --all`); GCP deploy story.
- `README.md` — add: landing-page section linking to MLflow + Prefect + Cloud Run URLs; quickstart for "train in Colab → deploy via PR"; Slack/CI/CD badge.
- `DOCS.md` — does not exist in this repo; do **not** create.
