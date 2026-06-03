# MLflow tracking server (B)

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
export GCP_PROJECT_ID=your-proj
export GCP_REGION=us-central1
export GAR_REPO=deepmaze
export MLFLOW_BUCKET=deepmaze-mlflow-artifacts
export SQL_INSTANCE=deepmaze-mlflow-sql
export SQL_PASSWORD='strong-secret'
export MLFLOW_SERVICE=mlflow-server

bash infra/mlflow/deploy.sh
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
