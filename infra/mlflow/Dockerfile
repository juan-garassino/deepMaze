FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
      mlflow==2.18.* \
      psycopg2-binary==2.9.* \
      google-cloud-storage==2.18.* \
      gunicorn==23.*

ENV PORT=8080 \
    BACKEND_STORE_URI=sqlite:////tmp/mlflow.db \
    ARTIFACT_ROOT=/tmp/mlartifacts

EXPOSE 8080

CMD exec mlflow server \
      --host 0.0.0.0 \
      --port ${PORT} \
      --backend-store-uri ${BACKEND_STORE_URI} \
      --default-artifact-root ${ARTIFACT_ROOT} \
      --serve-artifacts \
      --workers 2
