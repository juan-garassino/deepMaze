# infra/

Infrastructure-as-code for the MLOps subsystems. Each subdir is self-contained — provision them in any order you like, but the **suggested order** matches the build order from the spec: **B → C → D**.

| Dir | Subsystem | Purpose | Detail |
|---|---|---|---|
| [`mlflow/`](mlflow/) | B — MLflow server | Cloud Run + Cloud SQL + GCS artifact store | [`mlflow/README.md`](mlflow/README.md) |
| [`cloudrun/`](cloudrun/) | C — inference service | `service.yaml` rendered by GHA and applied via `gcloud run services replace` | inline header comment |
| [`prefect/`](prefect/) | D — local Prefect server | Optional Docker compose for running Prefect without Prefect Cloud | inline header comment |

## What's NOT here

- **GHA workflows** (E) live under `.github/workflows/` — not infra-as-code, but the same domain.
- **OTEL** (F) is in `web/otel.py` — application-level instrumentation, not infra.
- **Frontend deploy** is deferred (Firebase Hosting noted in the design spec but not provisioned).

## Conventions

- All `gcloud` commands accept `${GCP_PROJECT_ID}` + `${GCP_REGION}` from the environment.
- Compose files under `infra/` are **local dev only** — credentials are hardcoded and ports are bound to `localhost`.
- Scripts that mutate GCP (`mlflow/deploy.sh`) are idempotent — safe to re-run after a failure.

For the end-to-end runbook from a fresh GCP project, see [`../docs/deployment-guide.md`](../docs/deployment-guide.md).
