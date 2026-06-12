FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 + Claude Code (used only when CLAUDE_SELF_IMPROVE=true at runtime).
# Always installed because we want one image — toggle is runtime, not build-time.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt mlflow==2.18.*

COPY agents/      ./agents/
COPY environment/ ./environment/
COPY training/    ./training/
COPY utils/       ./utils/
COPY config/      ./config/
COPY scripts/train_runpod.py ./scripts/
COPY runpod/entrypoint.sh    ./
COPY runpod/program.md       ./

RUN chmod +x ./entrypoint.sh

# Git config so Claude can commit during self-improve iterations.
RUN git config --global user.email "claude-improve@deepmaze" && \
    git config --global user.name "claude-improve" && \
    git init && git add -A && git commit -m "initial baseline" --quiet

ENV OUTPUT_BASE=/workspace

ENTRYPOINT ["./entrypoint.sh"]
