#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deepMaze RunPod entrypoint
#
# Mirrors the autoresearch pattern (005-products/020-autoresearch/entrypoint.sh):
#   - Validate API key when self-improve is on
#   - Run the workload (training)
#   - Optionally hand control to Claude Code in --dangerously-skip-permissions
#     mode with a prompt file at /tmp/, logging to /app/claude.log
# ---------------------------------------------------------------------------
set -euo pipefail

# Use /workspace if RunPod mounted a volume; else /app/output.
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    OUTPUT_BASE="/workspace"
else
    OUTPUT_BASE="/app/output"
    mkdir -p "$OUTPUT_BASE"
fi
export OUTPUT_BASE

CLAUDE_SELF_IMPROVE="${CLAUDE_SELF_IMPROVE:-false}"
MAX_IMPROVE_ITERS="${MAX_IMPROVE_ITERS:-5}"
MAX_IMPROVE_HOURS="${MAX_IMPROVE_HOURS:-4}"

echo "=================================================================="
echo "  deepMaze RunPod training"
echo "  OUTPUT_BASE        = $OUTPUT_BASE"
echo "  CLAUDE_SELF_IMPROVE = $CLAUDE_SELF_IMPROVE"
echo "  MAX_IMPROVE_ITERS  = $MAX_IMPROVE_ITERS"
echo "  MAX_IMPROVE_HOURS  = $MAX_IMPROVE_HOURS"
echo "=================================================================="
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU detected — CPU run)"
echo ""

cd /app

# ---------------------------------------------------------------------------
# Phase 1 — baseline training (always runs).
# ---------------------------------------------------------------------------
echo "=== Phase 1: baseline training ==="
python scripts/train_runpod.py
echo "=== Baseline training complete ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 2 — optional Claude self-improvement loop.
# Toggle off by leaving CLAUDE_SELF_IMPROVE unset (default false).
# Same invocation shape as 020-autoresearch's entrypoint:
#   write prompt file under /tmp/, run claude --dangerously-skip-permissions
#   -p <prompt> --verbose, tee output to /app/claude.log
# ---------------------------------------------------------------------------
if [ "$CLAUDE_SELF_IMPROVE" != "true" ]; then
    echo "CLAUDE_SELF_IMPROVE != true — skipping improvement loop. Exiting."
    exit 0
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: CLAUDE_SELF_IMPROVE=true but ANTHROPIC_API_KEY is unset."
    echo "       Either: set CLAUDE_SELF_IMPROVE=false (train-only),"
    echo "           or: pass -e ANTHROPIC_API_KEY=sk-ant-... to docker run."
    exit 1
fi
export ANTHROPIC_API_KEY

# Branch each improve run for clean git history.
git checkout -b "claude-improve-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true

touch /app/claude.log

cat > /tmp/deepmaze_prompt.txt << 'PROMPT_EOF'
Please read /app/program.md for full context. Then execute the improvement
loop end-to-end:

  1. Read the baseline training results from ${OUTPUT_BASE}/mlruns/.
  2. Diagnose the failure mode in 3 bullets.
  3. Apply one targeted fix (start with the known structural bugs listed
     in program.md — per-step epsilon decay is the biggest one).
  4. Re-run training via `python scripts/train_runpod.py` (you can shrink
     CURRICULUM via env vars for faster iterations).
  5. Compare eval_success_rate; commit if improved, revert if not.
  6. Loop up to MAX_IMPROVE_ITERS times or until eval_success_rate > 0.9.
  7. At the end, write a markdown table summary of all iterations.

Begin now.
PROMPT_EOF

echo "=== Phase 2: Claude self-improvement loop ==="
echo "=== Prompt: ==="
cat /tmp/deepmaze_prompt.txt
echo ""
echo "=== Starting Claude Code (--dangerously-skip-permissions) ==="
echo ""

exec claude --dangerously-skip-permissions \
    -p "$(cat /tmp/deepmaze_prompt.txt)" \
    --verbose 2>&1 | tee /app/claude.log
