#!/usr/bin/env bash
# Provision + deploy MLflow tracking server on Cloud Run.
# Idempotent — safe to re-run.
#
# Required env (set in .env or your shell):
#   GCP_PROJECT_ID    — gcloud project
#   GCP_REGION        — e.g. europe-west1
#   GAR_REPO          — Artifact Registry repo name (will be created)
#   MLFLOW_BUCKET     — GCS bucket for artifacts (will be created)
#   SQL_INSTANCE      — Cloud SQL Postgres instance name (will be created)
#   SQL_DB            — Postgres database name (default: mlflow)
#   SQL_USER          — Postgres user (default: mlflow)
#   SQL_PASSWORD      — Postgres password (required, no default)
#   MLFLOW_SERVICE    — Cloud Run service name (default: mlflow-server)
set -euo pipefail

: "${GCP_PROJECT_ID:?required}"
: "${GCP_REGION:?required}"
: "${GAR_REPO:?required}"
: "${MLFLOW_BUCKET:?required}"
: "${SQL_INSTANCE:?required}"
: "${SQL_PASSWORD:?required}"
SQL_DB="${SQL_DB:-mlflow}"
SQL_USER="${SQL_USER:-mlflow}"
MLFLOW_SERVICE="${MLFLOW_SERVICE:-mlflow-server}"

IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GAR_REPO}/mlflow:latest"
SA_NAME="mlflow-server"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${GCP_PROJECT_ID}"

# --- APIs ---------------------------------------------------------------
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com

# --- Artifact Registry repo -------------------------------------------
gcloud artifacts repositories describe "${GAR_REPO}" --location="${GCP_REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${GAR_REPO}" \
       --repository-format=docker --location="${GCP_REGION}"

# --- GCS artifact bucket ----------------------------------------------
gcloud storage buckets describe "gs://${MLFLOW_BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${MLFLOW_BUCKET}" \
       --location="${GCP_REGION}" --uniform-bucket-level-access

# --- Cloud SQL Postgres ------------------------------------------------
gcloud sql instances describe "${SQL_INSTANCE}" >/dev/null 2>&1 \
  || gcloud sql instances create "${SQL_INSTANCE}" \
       --database-version=POSTGRES_15 \
       --tier=db-f1-micro \
       --region="${GCP_REGION}" \
       --storage-size=10GB \
       --storage-auto-increase

gcloud sql databases describe "${SQL_DB}" --instance="${SQL_INSTANCE}" >/dev/null 2>&1 \
  || gcloud sql databases create "${SQL_DB}" --instance="${SQL_INSTANCE}"

gcloud sql users list --instance="${SQL_INSTANCE}" --format="value(name)" | grep -q "^${SQL_USER}$" \
  || gcloud sql users create "${SQL_USER}" --instance="${SQL_INSTANCE}" --password="${SQL_PASSWORD}"

# --- Service account ---------------------------------------------------
gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA_NAME}" --display-name "MLflow Cloud Run"

gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/cloudsql.client" --condition=None >/dev/null

gcloud storage buckets add-iam-policy-binding "gs://${MLFLOW_BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin" >/dev/null

# --- Build + push image ------------------------------------------------
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
docker build -t "${IMAGE}" "$(dirname "$0")"
docker push "${IMAGE}"

# --- Deploy Cloud Run --------------------------------------------------
SQL_CONNECTION_NAME="${GCP_PROJECT_ID}:${GCP_REGION}:${SQL_INSTANCE}"
BACKEND_STORE_URI="postgresql://${SQL_USER}:${SQL_PASSWORD}@/${SQL_DB}?host=/cloudsql/${SQL_CONNECTION_NAME}"
ARTIFACT_ROOT="gs://${MLFLOW_BUCKET}"

gcloud run deploy "${MLFLOW_SERVICE}" \
  --image="${IMAGE}" \
  --region="${GCP_REGION}" \
  --platform=managed \
  --service-account="${SA_EMAIL}" \
  --add-cloudsql-instances="${SQL_CONNECTION_NAME}" \
  --set-env-vars="BACKEND_STORE_URI=${BACKEND_STORE_URI},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=2 \
  --allow-unauthenticated \
  --port=8080

URL=$(gcloud run services describe "${MLFLOW_SERVICE}" --region="${GCP_REGION}" --format="value(status.url)")
echo
echo "MLflow up at: ${URL}"
echo "Set MLFLOW_TRACKING_URI=${URL} in your notebook / backend / Prefect."
