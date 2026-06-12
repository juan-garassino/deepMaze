#!/usr/bin/env bash
# Submit a deepMaze training run as a Vertex AI custom job.
#
# No image build needed: uses Google's prebuilt PyTorch-GPU container,
# clones the (public) repo at startup and runs scripts/train_runpod.py.
# Artifacts land directly in the shared bucket via the automatic GCS FUSE
# mount (/gcs/<bucket>/...), so the Cloud Run backend can serve the bundles
# without any copy step. The GPU exists only for the job's lifetime —
# show-and-destroy is automatic.
#
# Usage:
#   bash scripts/train_vertex.sh                  # full memory-first curriculum
#   SMOKE=1 bash scripts/train_vertex.sh          # 5-min nano plumbing test
#   CURRICULUM="..." AGENTS_TO_RUN=dtqn bash scripts/train_vertex.sh
#
# Watch:
#   gcloud ai custom-jobs list --region=$REGION --project=$PROJECT
#   gcloud ai custom-jobs stream-logs <job-id> --region=$REGION

set -euo pipefail

PROJECT=${PROJECT:-garassino-ml}
REGION=${REGION:-europe-west1}
BRANCH=${BRANCH:-main}
BUCKET=${BUCKET:-garassino-ml-artifacts}
MACHINE=${MACHINE:-n1-standard-4}
GPU_TYPE=${GPU_TYPE:-NVIDIA_TESLA_T4}
GPU_COUNT=${GPU_COUNT:-1}
IMAGE=${IMAGE:-europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest}

TAG=$(date -u +%Y%m%d-%H%M%S)
RUN_TAG=${RUN_TAG:-v$TAG}

if [ "${SMOKE:-0}" = "1" ]; then
  NAME="deepmaze-smoke-$TAG"
  TRAIN_ENV=$(cat <<EOF
        - name: NANO
          value: "true"
        - name: AGENTS_TO_RUN
          value: "drqn"
        - name: CURRICULUM
          value: "8,8,1,60,40"
        - name: PARTIAL
          value: "2"
EOF
)
else
  NAME="deepmaze-train-$TAG"
  TRAIN_ENV=$(cat <<EOF
        - name: AGENTS_TO_RUN
          value: "${AGENTS_TO_RUN:-drqn}"
        - name: CURRICULUM
          value: "${CURRICULUM:-10,20,2,800,180,8;20,40,4,1500,720,16;30,60,6,2500,1620,32}"
        - name: ADVANCE_THRESHOLD
          value: "${ADVANCE_THRESHOLD:-0.5}"
        - name: STAGE_MAX_REPEATS
          value: "${STAGE_MAX_REPEATS:-2}"
EOF
)
fi

CFG=$(mktemp /tmp/deepmaze-vertex-XXXXXXXX).yaml
cat > "$CFG" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
      acceleratorType: ${GPU_TYPE}
      acceleratorCount: ${GPU_COUNT}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE}
      command: ["bash", "-c"]
      args:
        - |
          set -e
          git clone --depth 1 -b ${BRANCH} https://github.com/juan-garassino/deepMaze.git /app
          cd /app
          pip install -q -r requirements.txt mlflow
          python scripts/train_runpod.py
      env:
        - name: OUTPUT_BASE
          value: "/gcs/${BUCKET}/deepmaze/vertex/${RUN_TAG}"
        - name: RUN_TAG
          value: "${RUN_TAG}"
${TRAIN_ENV}
EOF

echo "Submitting ${NAME} (${MACHINE} + ${GPU_COUNT}x${GPU_TYPE}, branch ${BRANCH})"
echo "Artifacts → gs://${BUCKET}/deepmaze/vertex/${RUN_TAG}/"
gcloud ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --config="${CFG}"
