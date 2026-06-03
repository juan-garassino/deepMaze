# Deployment guide

Cold-start runbook: from an empty GCP project to a deployed inference service + MLflow + Slack-wired CI/CD. Steps are sequential; each one is idempotent.

## 0. Prerequisites

You need:
- A GCP project with billing enabled. (Reserve ~$5–10/month for Cloud SQL `db-f1-micro` + Cloud Run minScale=0.)
- A GitHub repo (this one) — Actions enabled.
- A Slack workspace where you can create an incoming webhook.
- Local tools: `gcloud`, `docker`, `gh`, `git`.

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

GitHub Actions authenticates to GCP via OIDC. One-time setup:

```bash
# Pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" --display-name="GitHub Actions pool"

# Provider, restricted to your GitHub user/org
GH_USER=juan-garassino   # change to your GH username or org
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --workload-identity-pool="github-pool" --location="global" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner=='${GH_USER}'"

# Service account that GHA will impersonate
gcloud iam service-accounts create "github-deployer" \
  --display-name="GHA deploy SA"
SA_EMAIL="github-deployer@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# GHA → SA impersonation binding
PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')
WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
GH_REPO=juan-garassino/deepMaze
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WIF_PROVIDER/projects\/${PROJECT_NUMBER}/projects\/${PROJECT_NUMBER}}/attribute.repository/${GH_REPO}"

# SA permissions for deploy
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" --role="${role}" --condition=None
done
```

Record `${WIF_PROVIDER}` and `${SA_EMAIL}` — they go into GitHub secrets in step 4.

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
2. Set Colab form fields:
   - `MLFLOW_TRACKING_URI` = the URL from step 3
   - `ASSETS_BUCKET` = `${ASSETS_BUCKET}` from step 4
   - Upload your GCS key JSON via the Colab file pane (left sidebar). Set `GOOGLE_APPLICATION_CREDENTIALS=/content/<your-key>.json` in a fresh cell.
3. **Runtime → Change runtime type → GPU** (T4 is enough for DRQN; A100 for DTQN with batch >32).
4. **Runtime → Run all**.
5. Verify in the MLflow UI: open `${MLFLOW_TRACKING_URI}` → experiment `deepmaze` → your run with `eval_success_rate` metric.
6. Either:
   - **Direct GCS push** (the notebook did this if `ASSETS_BUCKET` was set) — the bundle is already at `gs://${ASSETS_BUCKET}/${RUN_NAME}/`.
   - **Local download** — unzip the `.zip` from the Colab file pane into `assets/<run_name>/` and `git push` it.

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

curl -s -X POST "${CLOUD_RUN_URL}/api/inference" \
  -H "Content-Type: application/json" \
  -d '{"source":"asset","name":"drqn_v1","maze_source":"trained"}' | head -20
```

You should see SSE `data: {...}` lines ending in `data: {"type":"episode","done":true,...}`.

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

## What's intentionally NOT covered

Per the design spec, these are deferred:
- Auth on the public `/api/inference` (front with IAP for real deploys)
- Vertex AI training (Colab first)
- Multi-region Cloud Run
- Cost dashboards
- Firebase Hosting frontend split (currently bundled with backend at `/static/`)
