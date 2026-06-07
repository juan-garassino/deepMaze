# MLflow tracking server (B) — REFERENCE ONLY

> ⚠️ **This folder is not used by the live deepMaze pipeline.** Under the post-2026-06-07 architecture (`garassino-ml` / €25/mo cap / no Cloud SQL), MLflow is **file://** everywhere — Drive on Colab, `/workspace/mlruns/` on RunPod, `./local_runs/mlruns/` locally. The Cloud Run + Cloud SQL + GCS recipe below is kept as **prior art** for the day the project needs a long-lived tracking server again; if so, swap Cloud SQL for Neon free tier per workspace policy. The `deploy.sh` here is **not** wired into any GitHub Actions workflow.

Two deploy modes: **local dev** (Docker) and **GCP** (Cloud Run + Cloud SQL + GCS).

## Local dev (Docker)

```bash
cd infra/mlflow
docker compose -f docker-compose.local.yml up --build
# → http://localhost:5000  (host:5000 → container:8080)
```

Use `MLFLOW_TRACKING_URI=http://localhost:5000` in the notebook and the backend. On Cloud Run the server listens on port 8080 internally; the public URL has no port.

## GCP (Cloud Run)

Set the required env then run the deploy script:

```bash
export GCP_PROJECT_ID=garassino-ml          # post-2026-06-07 target
export GCP_REGION=europe-west1
export GAR_REPO=mlflow                      # only if you really revive the GAR path
export MLFLOW_BUCKET=garassino-ml-artifacts # shared bucket; subpaths per project
export SQL_INSTANCE=deepmaze-mlflow-sql     # Cloud SQL — excluded by workspace policy
export SQL_PASSWORD='strong-secret'
export MLFLOW_SERVICE=mlflow-server

bash infra/mlflow/deploy.sh --force          # --force required — script is guarded
```

The script is idempotent: it creates only what doesn't already exist (Artifact Registry repo, GCS bucket, Cloud SQL instance + db + user, service account + bindings), builds + pushes the image, and deploys the Cloud Run revision. It prints the public URL at the end — that's your `MLFLOW_TRACKING_URI`.

## Auth model

| Caller | Auth |
|---|---|
| Cloud Run → Cloud SQL | Unix socket at `/cloudsql/<conn>` via `roles/cloudsql.client` |
| Cloud Run → GCS | Workload identity via `roles/storage.objectAdmin` on the bucket |
| Notebook → MLflow REST | Public (no auth in this phase — out of scope per spec) |
| Notebook → GCS for artifact upload | `GOOGLE_APPLICATION_CREDENTIALS` JSON key with `roles/storage.objectAdmin` on the bucket |

> **Security note.** This is dev-grade: the tracking server is unauthenticated, and `BACKEND_STORE_URI` embeds the SQL password in plaintext as a Cloud Run env var (visible in the GCP console). Before any real deploy: front the service with IAP, and move the SQL password into Secret Manager (`gcloud run deploy --set-secrets BACKEND_STORE_URI=mlflow-uri:latest`).

## What the server exposes

- `GET  /api/2.0/mlflow/...` — MLflow REST API
- `GET  /` — UI
- artifact proxy (we pass `--serve-artifacts`) so clients don't need direct GCS access for read

## Verifying

```bash
curl -s "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/search" \
  -X POST -H "Content-Type: application/json" -d '{"max_results":1}'
```

Should return `{"experiments":[...]}` (possibly empty).
