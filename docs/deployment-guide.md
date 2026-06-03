# Deployment guide

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
export GCP_PROJECT_ID=deepmaze-prod      # whatever you named it
export GCP_REGION=us-central1
gcloud config set project "${GCP_PROJECT_ID}"
```

Enable APIs (the MLflow deploy script does this too, but front-loading it surfaces quota issues earlier):

```bash
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
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

## 3. Provision MLflow

```bash
export GAR_REPO=deepmaze
export MLFLOW_BUCKET="${GCP_PROJECT_ID}-mlflow-artifacts"
export SQL_INSTANCE=deepmaze-mlflow-sql
export SQL_PASSWORD="$(openssl rand -base64 24)"
echo "Save SQL_PASSWORD safely: ${SQL_PASSWORD}"
export MLFLOW_SERVICE=mlflow-server

bash infra/mlflow/deploy.sh
```

When it finishes, copy the printed URL — that's your `MLFLOW_TRACKING_URI`.

> Public + unauthenticated by default per the spec. Front with IAP before any non-toy deploy.

## 4. Create the assets bucket + Cloud Run runtime service account

```bash
export ASSETS_BUCKET="${GCP_PROJECT_ID}-deepmaze-assets"
gcloud storage buckets create "gs://${ASSETS_BUCKET}" \
  --location="${GCP_REGION}" --uniform-bucket-level-access

# Runtime SA used by the inference Cloud Run service
RUNTIME_SA="deepmaze-runtime"
gcloud iam service-accounts create "${RUNTIME_SA}" --display-name="deepMaze runtime"
RUNTIME_SA_EMAIL="${RUNTIME_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://${ASSETS_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
  --role="roles/storage.objectViewer"

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
| `GCP_PROJECT_ID` | `${GCP_PROJECT_ID}` |
| `WIF_PROVIDER` | the full provider resource path from step 2 |
| `WIF_SERVICE_ACCOUNT` | `github-deployer@${GCP_PROJECT_ID}.iam.gserviceaccount.com` |
| `GAR_REPO` | `deepmaze` |
| `CLOUD_RUN_SERVICE` | `deepmaze-backend` |
| `CLOUD_RUN_SA_EMAIL` | `${RUNTIME_SA_EMAIL}` |
| `SLACK_WEBHOOK_URL` | the Slack webhook from step 5 |

**Repository variables:**

| Name | Value |
|---|---|
| `GCP_REGION` | `us-central1` |
| `ASSETS_BUCKET` | `${ASSETS_BUCKET}` |
| `CORS_ORIGINS` | the deployed frontend origin, or `*` for demo |

## 7. First Colab training run

1. Open `notebooks/train_agent.ipynb` in Colab (`File → Open notebook → GitHub → juan-garassino/deepMaze → notebooks/train_agent.ipynb`).
2. **Runtime → Change runtime type → GPU** (T4 is enough for DRQN; A100 helps DTQN with batch > 32).
3. Set the Colab form fields in cell 1:
   - `MLFLOW_TRACKING_URI` = the URL from step 3
   - `ASSETS_BUCKET` = `${ASSETS_BUCKET}` from step 4 (leave blank to skip GCS push and download a zip instead)
4. If `ASSETS_BUCKET` is set, give Colab GCS write access. Add this cell **after** the config cell and run it once:
   ```python
   from google.colab import files
   import os
   uploaded = files.upload()           # pick your SA JSON key
   key = "/content/" + next(iter(uploaded))
   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key
   print("GAC set to:", key)
   ```
   The key gets a Colab-local path and stays out of the `.ipynb` source.
5. **Runtime → Run all**.
6. Verify in the MLflow UI: open `${MLFLOW_TRACKING_URI}` → experiment `deepmaze` → your run with `eval_success_rate` logged.
7. The bundle lands in one of two places:
   - **GCS** (if `ASSETS_BUCKET` was set) — at `gs://${ASSETS_BUCKET}/${RUN_NAME}/`. The deployed backend syncs it on next instance start.
   - **Colab `/content/${RUN_NAME}.zip`** — download via the file pane, unzip into `assets/<run_name>/`, `git add` + push.

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
