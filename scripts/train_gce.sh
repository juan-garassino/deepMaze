#!/usr/bin/env bash
# Train deepMaze on a Compute Engine SPOT T4 VM (the project has 1x T4 GCE
# quota in europe-west1; Vertex custom-training GPU quota is 0 until a
# support request lands — this path needs nothing).
#
# Lifecycle: boots a deep-learning VM image (CUDA + PyTorch preinstalled),
# clones the repo, trains via scripts/train_runpod.py, rsyncs artifacts to
# GCS every 5 minutes (spot-preemption protection) and once at the end,
# then DELETES ITSELF. Cost: ~$0.15/hr while training, zero after.
#
# Usage:
#   bash scripts/train_gce.sh                # full memory-first curriculum
#   SMOKE=1 bash scripts/train_gce.sh        # nano plumbing test (~10 min)
#
# Watch:
#   gcloud compute instances list --project=garassino-ml
#   gcloud compute ssh <name> --zone=$ZONE -- tail -f /var/log/syslog
#   gsutil ls gs://garassino-ml-artifacts/deepmaze/gce/<run-tag>/

set -euo pipefail

PROJECT=${PROJECT:-garassino-ml}
ZONE=${ZONE:-europe-west1-b}
BRANCH=${BRANCH:-main}
BUCKET=${BUCKET:-garassino-ml-artifacts}
MACHINE=${MACHINE:-n1-standard-4}

TAG=$(date -u +%Y%m%d-%H%M%S)
RUN_TAG=${RUN_TAG:-v$TAG}
GCS_DEST="gs://${BUCKET}/deepmaze/gce/${RUN_TAG}"

if [ "${SMOKE:-0}" = "1" ]; then
  NAME="deepmaze-smoke-$TAG"
  TRAIN_VARS='export NANO=true AGENTS_TO_RUN=drqn CURRICULUM="8,8,1,60,40" PARTIAL=2'
else
  NAME="deepmaze-train-$TAG"
  TRAIN_VARS="export AGENTS_TO_RUN=${AGENTS_TO_RUN:-drqn} \
CURRICULUM=\"${CURRICULUM:-10,20,2,800,180,8;20,40,4,1500,720,16;30,60,6,2500,1620,32}\" \
ADVANCE_THRESHOLD=${ADVANCE_THRESHOLD:-0.5} STAGE_MAX_REPEATS=${STAGE_MAX_REPEATS:-2}"
fi

STARTUP=$(mktemp /tmp/deepmaze-gce-XXXXXXXX).sh
cat > "$STARTUP" <<STARTUP_EOF
#!/bin/bash
set -x
exec > >(tee /var/log/deepmaze-train.log) 2>&1

# DLVM installs the NVIDIA driver on first boot when asked via metadata;
# wait for it.
for i in \$(seq 1 60); do nvidia-smi && break; sleep 10; done

git clone --depth 1 -b ${BRANCH} https://github.com/juan-garassino/deepMaze.git /opt/deepMaze
cd /opt/deepMaze
/opt/conda/bin/pip install -q -r requirements.txt mlflow

export OUTPUT_BASE=/opt/dm-out
${TRAIN_VARS}
export RUN_TAG=${RUN_TAG}

# spot-preemption protection: sync partial artifacts every 5 min
( while true; do sleep 300; gsutil -m -q rsync -r /opt/dm-out ${GCS_DEST} || true; done ) &
SYNC_PID=\$!

/opt/conda/bin/python scripts/train_runpod.py
STATUS=\$?

kill \$SYNC_PID || true
gsutil -m -q rsync -r /opt/dm-out ${GCS_DEST} || true
gsutil cp /var/log/deepmaze-train.log ${GCS_DEST}/train.log || true
echo "training exited \$STATUS — artifacts at ${GCS_DEST} — deleting instance"
gcloud compute instances delete ${NAME} --zone=${ZONE} --quiet
STARTUP_EOF

echo "Creating SPOT VM ${NAME} (${MACHINE} + 1x T4, zone ${ZONE})"
echo "Artifacts → ${GCS_DEST}/"
gcloud compute instances create "${NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MACHINE}" \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580} --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script="${STARTUP}"

echo
echo "Tail progress:  gcloud compute ssh ${NAME} --zone=${ZONE} --project=${PROJECT} -- tail -f /var/log/deepmaze-train.log"
echo "Kill early:     gcloud compute instances delete ${NAME} --zone=${ZONE} --project=${PROJECT}"
