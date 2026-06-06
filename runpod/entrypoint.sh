#!/usr/bin/env bash
set -euo pipefail

# Use /workspace if RunPod mounted a volume; else /app/output.
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    OUTPUT_BASE="/workspace"
else
    OUTPUT_BASE="/app/output"
    mkdir -p "$OUTPUT_BASE"
fi

export OUTPUT_BASE
echo "=== deepMaze RunPod training ==="
echo "OUTPUT_BASE = $OUTPUT_BASE"
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU detected — CPU run)"
echo ""

cd /app
exec python scripts/train_runpod.py
