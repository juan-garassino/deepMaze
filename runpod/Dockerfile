FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Repo deps (cached when requirements.txt is unchanged)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt mlflow==2.18.*

# Repo source (everything the training script imports)
COPY agents/      ./agents/
COPY environment/ ./environment/
COPY training/    ./training/
COPY utils/       ./utils/
COPY config/      ./config/
COPY scripts/train_runpod.py ./scripts/
COPY runpod/entrypoint.sh    ./

RUN chmod +x ./entrypoint.sh

# Outputs land in /workspace (RunPod's standard persistent volume mount).
# If no volume is mounted, falls back to /app/output.
ENV OUTPUT_BASE=/workspace

ENTRYPOINT ["./entrypoint.sh"]
