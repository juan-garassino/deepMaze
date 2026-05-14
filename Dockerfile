# deepMaze backend: FastAPI + torch (CPU) + agents.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Pin torch from the CPU wheel index — keeps the image under 1 GB.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.* \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# torch already installed above; drop it from this file's resolution.
RUN grep -v '^torch' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

COPY . .

EXPOSE 8000
ENV PYTHONPATH=/app
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
