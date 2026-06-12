# Deployment guide

> **Pre-migration doc (2026-06-07):** GCP target is now **`garassino-ml`** / `europe-west1` in show-and-destroy mode under €25/mo workspace cap. Region/project references below describe pre-migration state. Canonical config in workspace root `CLAUDE.md` § "GCP architecture".

Cold-start runbook: from an empty GCP project to a deployed inference service + MLflow + Slack-wired CI/CD. Steps are sequential; each one is idempotent.

## 0. Prerequisites

You need:
- A GCP project with billing enabled.
- A GitHub repo (this one) — Actions enabled.
- A Slack workspace where you can create an incoming webhook.
- Local tools: `gcloud`, `docker`, `gh`, `git`.

**Cost at rest** (no traffic, idle MLflow): Cloud SQL `db-f1-micro` is the floor at **~$8/month**. Add Cloud Run minScale=0 (~$0), Artifact Registry (~$0), GCS storage (~$0 at this scale), egress (~$0 for demo traffic) → realistically **$10–15/month** as a learning project. Pause everything via the tear-down section at the end of this doc when not in use.

Authenticate locally once:
```bash
gcloud auth login
gcloud auth application-default login
gh auth login
```

## 1. One-time GCP project setup

Set the project + region you'll use throughout:

```bash
export GCP_PROJECT_ID=garassino-ml        # post-2026-06-07: shared ML project
export GCP_REGION=europe-west1            # post-2026-06-07: consolidated region
gcloud config set project "${GCP_PROJECT_ID}"
```

Enable APIs (Cloud SQL + Artifact Registry are no longer needed under the new architecture — MLflow is file:// everywhere and images live on GHCR):

```bash
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudtrace.googleapis.com
```

## 2. Workload Identity Federation (GHA → GCP, no long-lived keys)

GitHub Actions authenticates to GCP via OIDC — no JSON service-account keys to rotate. One-time setup:

```bash
GH_USER=juan-garassino                     # your GH username or org
GH_REPO=juan-garassino/deepMaze            # the repo GHA runs from

PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')
WIF_POOL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool"

# 2a. Pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" --display-name="GitHub Actions pool"

# 2b. Provider, restricted to your GitHub user/org so other repos can't impersonate
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --workload-identity-pool="github-pool" --location="global" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner=='${GH_USER}'"

WIF_PROVIDER="${WIF_POOL}/providers/github-provider"

# 2c. Service account that GHA will impersonate
gcloud iam service-accounts create "github-deployer" --display-name="GHA deploy SA"
SA_EMAIL="github-deployer@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# 2d. Allow GHA from this specific repo to impersonate the SA
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WIF_POOL}/attribute.repository/${GH_REPO}"

# 2e. SA permissions for deploy
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" --role="${role}" --condition=None
done

echo
echo "Copy these into GitHub repo secrets (step 6):"
echo "  WIF_PROVIDER         = ${WIF_PROVIDER}"
echo "  WIF_SERVICE_ACCOUNT  = ${SA_EMAIL}"
```

The `principalSet://...` member binds to the **pool** (not the provider) so any provider under the pool can impersonate — the per-provider `--attribute-condition` is what actually scopes access to your repo.

## 3. Provision MLflow — SKIP under the new architecture

> **Skip this step.** Post-2026-06-07, MLflow runs as a file:// store everywhere — Drive on Colab, `/workspace/mlruns/` on RunPod, `./local_runs/mlruns/` locally. No Cloud Run + Cloud SQL server. The `infra/mlflow/deploy.sh` script is **reference only** (guarded with `--force` to prevent accidental runs). If you ever need a persistent tracking server, swap Cloud SQL for Neon free tier per workspace policy.

## 4. Storage layout + Cloud Run runtime service account

Under the new architecture, **all deep-* projects share one bucket** (`garassino-ml-artifacts`) with per-project prefixes. deepMaze owns the `deepmaze/` prefix.

```bash
# Shared bucket — created once for the workspace, used by every deep-* project.
export ASSETS_BUCKET="garassino-ml-artifacts"
export ASSETS_PREFIX="deepmaze/"   # this project's subpath inside the shared bucket
gcloud storage buckets create "gs://${ASSETS_BUCKET}" \
  --location="${GCP_REGION}" --uniform-bucket-level-access 2>/dev/null || \
  echo "Bucket already exists — fine, that's the point of the shared layout."

# Runtime SA used by the inference Cloud Run service
RUNTIME_SA="deepmaze-runtime"
gcloud iam service-accounts create "${RUNTIME_SA}" --display-name="deepMaze runtime"
RUNTIME_SA_EMAIL="${RUNTIME_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Read-only access to just deepMaze's prefix (not the whole shared bucket).
gcloud storage buckets add-iam-policy-binding "gs://${ASSETS_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
  --role="roles/storage.objectViewer" \
  --condition="title=deepmaze-prefix-only,expression=resource.name.startsWith('projects/_/buckets/${ASSETS_BUCKET}/objects/${ASSETS_PREFIX}')"

gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
  --role="roles/cloudtrace.agent" --condition=None
```

## 5. Slack webhook

In Slack: **Apps → Incoming Webhooks → Add to Slack → pick a channel → copy the webhook URL**.

## 6. GitHub repo secrets + vars

In **Settings → Secrets and variables → Actions**:

**Repository secrets:**

| Name | Value |
|---|---|
| `GCP_PROJECT_ID` | `garassino-ml` |
| `WIF_PROVIDER` | the full provider resource path from `garassino-op`'s WIF pool (workspace policy: WIF lives in `garassino-op`, not here) |
| `WIF_SERVICE_ACCOUNT` | the runtime SA email that the WIF pool can impersonate |
| `CLOUD_RUN_SERVICE` | `deepmaze-backend` |
| `CLOUD_RUN_SA_EMAIL` | `${RUNTIME_SA_EMAIL}` |
| `SLACK_WEBHOOK_URL` | the Slack webhook from step 5 (optional) |
| `TELEGRAM_BOT_TOKEN` | from @BotFather (optional, see CLAUDE.md § "Telegram notifications") |
| `TELEGRAM_CHAT_ID` | your chat id (optional) |

> `GAR_REPO` is **no longer required** — images live on GHCR (`ghcr.io/juan-garassino/deepmaze-backend`); the workflow uses the built-in `GITHUB_TOKEN` to push.

**Repository variables:**

| Name | Value |
|---|---|
| `GCP_REGION` | `europe-west1` |
| `ASSETS_BUCKET` | `garassino-ml-artifacts` |
| `ASSETS_PREFIX` | `deepmaze/` |
| `CORS_ORIGINS` | the deployed frontend origin, or `*` for demo |

## 7. First Colab training run

Under the new architecture the notebook is **Drive-backed** — no Cloud MLflow, no GCS SA key in Colab. Bundles land in Drive; you copy them to the shared `garassino-ml-artifacts` bucket from your laptop with `gsutil` (or via a future GHA workflow).

1. Open `notebooks/train_agent.ipynb` in Colab (`File → Open notebook → GitHub → juan-garassino/deepMaze → notebooks/train_agent.ipynb`).
2. **Runtime → Change runtime type → GPU** (T4 is enough for DRQN; A100 helps DTQN with batch > 32).
3. Set the form fields in cell 4. Defaults are fine for a first run; `IS_COLAB` is auto-detected in cell 2 and the notebook switches to Drive-backed mode automatically.
4. **Runtime → Run all**.
5. The bundle lands at `${DRIVE_BASE}/assets/<run_name>/` (defaults to `/content/drive/MyDrive/deepMaze/assets/...`). MLflow runs are next to it at `${DRIVE_BASE}/mlruns/`.
6. From your laptop, after Drive sync settles, push the bundle to GCS so the Cloud Run backend can sync it:
   ```bash
   gsutil -m cp -r ~/Drive/MyDrive/deepMaze/assets/<run_name> \
     gs://garassino-ml-artifacts/deepmaze/<run_name>
   ```
7. Restart the Cloud Run revision (`gcloud run services update-traffic deepmaze-backend --to-latest --region europe-west1`) so the entrypoint re-runs `sync_assets.py` and pulls the new bundle.

## 8. First deploy

Push to `main` (anything — a comment, a doc fix, or just the asset commit from step 7):

```bash
git push origin main
```

Watch:
- GitHub Actions tab → `deploy` workflow turning green at each step.
- Slack channel → "deepMaze deploy started" then "succeeded".
- Cloud Run console → `deepmaze-backend` with a new revision.

## 9. Verify with a smoke test

Manually:

```bash
CLOUD_RUN_URL=$(gcloud run services describe deepmaze-backend \
  --region "${GCP_REGION}" --format='value(status.url)')

curl -sN -X POST "${CLOUD_RUN_URL}/api/inference" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"source":"asset","name":"drqn_v1","maze_source":"trained"}' | head -20
```

The `-N` flag keeps curl unbuffered so SSE lines stream as they arrive. You should see `data: {...}` lines ending in `data: {"type":"episode","done":true,...}`.

> First request after a fresh deploy: the asset sync runs in the background, so a 404 in the first 5 s is expected — retry in 10 s.

Via Prefect:

```bash
prefect work-pool create --type process default-process    # one-time
prefect deploy --all --prefect-file flows/prefect.yaml
CLOUD_RUN_URL="${CLOUD_RUN_URL}" prefect deployment run 'smoke_test_flow/smoke-test'
```

## 10. Wire Prefect for retraining (optional)

Sign up for [Prefect Cloud](https://app.prefect.cloud) (free tier). Then:

```bash
prefect cloud login
prefect work-pool create --type process default-process
prefect deploy --all --prefect-file flows/prefect.yaml
```

From now on:
- `retrain_flow` watches MLflow for new champion runs and calls `promote_flow` automatically.
- `promote_flow <run_id>` can be triggered manually after any Colab session.
- `smoke_test_flow` runs daily.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| GHA deploy fails on `auth` step | `WIF_PROVIDER` resource path malformed | re-copy from step 2; check the `${GH_REPO}` matches your repo name exactly |
| GHA deploy succeeds but Cloud Run revision unhealthy | `gs://${ASSETS_BUCKET}` empty | promote at least one model bundle first (step 7) |
| `/api/inference` 404 | asset name not under `/app/assets/` | check `gsutil ls gs://${ASSETS_BUCKET}/` and wait ~10 s for the background sync after a fresh instance |
| Notebook crashes on MLflow log | `MLFLOW_TRACKING_URI` unreachable from Colab | Colab can hit any public URL; if you're using a local server, expose via `ngrok http 5000` |
| `prefect deploy` fails with "no work pool" | `default-process` not created | run `prefect work-pool create --type process default-process` first |
| MLflow Cloud Run 502 on artifact download | bucket SA binding missing | re-run `infra/mlflow/deploy.sh` (idempotent) — it re-adds the storage binding |

## Tear-down

To stop the bill cold, remove resources in reverse-create order. All commands are idempotent; safe to re-run:

```bash
# Cloud Run services
gcloud run services delete deepmaze-backend --region "${GCP_REGION}" --quiet
gcloud run services delete mlflow-server    --region "${GCP_REGION}" --quiet

# Cloud SQL — the biggest line item
gcloud sql instances delete "${SQL_INSTANCE}" --quiet

# Storage buckets
gcloud storage rm -r "gs://${ASSETS_BUCKET}"  --quiet
gcloud storage rm -r "gs://${MLFLOW_BUCKET}"  --quiet

# Artifact Registry repo (drops all images)
gcloud artifacts repositories delete "${GAR_REPO}" --location="${GCP_REGION}" --quiet

# Service accounts + IAM
for sa in github-deployer mlflow-server deepmaze-runtime; do
  gcloud iam service-accounts delete "${sa}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" --quiet
done

# Workload Identity pool
gcloud iam workload-identity-pools delete github-pool --location=global --quiet
```

To **pause without destroying** (keeps the SQL data, image, and assets — restart in minutes):

```bash
gcloud sql instances patch "${SQL_INSTANCE}" --activation-policy=NEVER
gcloud run services update mlflow-server    --region "${GCP_REGION}" --max-instances=0
gcloud run services update deepmaze-backend --region "${GCP_REGION}" --max-instances=0
```

Resume with `--activation-policy=ALWAYS` and `--max-instances=2`.

## Common operations

| Task | Command |
|---|---|
| View current Cloud Run revision | `gcloud run services describe deepmaze-backend --region "${GCP_REGION}"` |
| Roll back to previous revision | `gcloud run services update-traffic deepmaze-backend --region "${GCP_REGION}" --to-revisions=<prev>=100` |
| Tail backend logs | `gcloud run services logs read deepmaze-backend --region "${GCP_REGION}" --limit 50` |
| Tail MLflow logs | `gcloud run services logs read mlflow-server --region "${GCP_REGION}" --limit 50` |
| List promoted assets in the bucket | `gsutil ls "gs://${ASSETS_BUCKET}/"` |
| Trigger a re-deploy without code change | `git commit --allow-empty -m "redeploy" && git push` |
| Rotate the Slack webhook | regenerate in Slack, update `SLACK_WEBHOOK_URL` repo secret |
| Rotate the SQL password | `gcloud sql users set-password mlflow --instance="${SQL_INSTANCE}" --password=NEW` then `export SQL_PASSWORD=NEW && bash infra/mlflow/deploy.sh` to refresh the `BACKEND_STORE_URI` env on the Cloud Run revision |

## Local-only development (no GCP)

You can exercise the full pipeline without touching GCP — useful for iterating on the notebook or flows:

```bash
# 1. Local MLflow + Postgres
docker compose -f infra/mlflow/docker-compose.local.yml up -d --build
# MLflow at http://localhost:5000

# 2. Run the notebook against http://localhost:5000 (Colab reaches it via ngrok,
#    or use a local Jupyter kernel — see notebooks/README.md)

# 3. Local backend with the trained bundle
docker compose up --build
# Frontend http://localhost:8080, backend http://localhost:8000

# 4. Local Prefect (optional)
docker compose -f infra/prefect/docker-compose.yml up -d
prefect work-pool create --type process default-process
prefect deploy --all --prefect-file flows/prefect.yaml
```

This path skips deploy.yml entirely — no Slack notifications, no Cloud Run.

## What's intentionally NOT covered

Per the design spec, these are deferred:
- Auth on the public `/api/inference` (front with IAP for real deploys)
- Vertex AI training (Colab first)
- Multi-region Cloud Run
- Cost dashboards / budget alerts
- Firebase Hosting frontend split (currently bundled with backend at `/static/`)
- Secret Manager for SQL password (currently plain env var on the MLflow Cloud Run service)
