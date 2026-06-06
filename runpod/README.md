# RunPod training

Standalone GPU training for deepMaze. Runs the same `train_one` + curriculum logic as the notebook, but as a single `python scripts/train_runpod.py` invocation reading config from env vars.

## Files

| File | What it does |
|---|---|
| `Dockerfile` | CUDA 12.1 + PyTorch 2.2 image with the repo + entrypoint baked in. |
| `entrypoint.sh` | Detects `/workspace` (RunPod volume) → falls back to `/app/output` → `exec python scripts/train_runpod.py`. |
| (root) `Makefile` | `make build / push / run / logs / stop` — mirrors the autoresearch pattern. |
| (root) `scripts/train_runpod.py` | The standalone training driver. All knobs are env vars. |

## Build + push

```bash
make build                       # builds runpod/Dockerfile → deepmaze-train
make push REGISTRY=garassinoj    # tags + pushes to Docker Hub
```

## On RunPod

1. **Create a GPU Pod** — T4 enough for DRQN, A100/H100 for DTQN with big sequences.
2. **Container image:** `garassinoj/deepmaze-train` (whatever you pushed).
3. **Volume:** mount one at `/workspace` — that's where MLflow + bundles get written. Without a volume, outputs land in `/app/output` and disappear when the pod stops.
4. **Environment variables** (all optional, defaults match the notebook "real run"):

   | Var | Default | Notes |
   |---|---|---|
   | `AGENTS_TO_RUN` | `drqn,dtqn` | Comma-separated. |
   | `RUN_TAG` | `v1` | Suffix on run names. |
   | `CURRICULUM` | `10,20,5,800,200;20,40,8,1200,400;30,60,10,2000,600` | Semicolon-separated `W,H,n_treasures,episodes,max_steps` tuples. |
   | `MAZE_WIDTH` etc. | unset | If set, **bypasses curriculum** — runs a single stage at the given dims. |
   | `PARTIAL` | `5` | Egocentric window radius. Must stay constant across curriculum stages for weight transfer. |
   | `EXPLORATION_DECAY` | `0.999995` | Per-step ε decay. The repo's per-step (not per-episode) decay needs this slow on big mazes. |
   | `BUFFER_CAPACITY` | `50000` | Replay buffer transitions. |
   | `REGENERATE_EVERY` | `1` | Re-roll maze each episode (generalization). 0 = off (memorize one maze). |
   | `EVAL_EPISODES` | `50` | Held-out eval episodes after each stage. |
   | `OUTPUT_BASE` | `/workspace` | Set by entrypoint; override if you need a different mount. |

5. **Start the pod** — training begins automatically. Tail the logs from RunPod's UI.

## Outputs (per stage)

```
${OUTPUT_BASE}/
  mlruns/                          # full MLflow file store
  assets/<agent>_<tag>_s<i>_WxH/   # bundle: config.json + model.pt + viz/replay.webp
  showcase/<run_name>/             # periodic greedy replays during training
```

Copy `assets/<name>/` back to your laptop and drop into `<repo>/assets/<name>/` to use with `python web/server.py`.

## Local smoke-test (without Docker)

```bash
make local
# or, equivalently:
OUTPUT_BASE=$PWD/local_runs \
AGENTS_TO_RUN=drqn \
CURRICULUM="8,8,1,200,100" \
EXPLORATION_DECAY=0.99 BUFFER_CAPACITY=2000 PARTIAL=2 \
python scripts/train_runpod.py
```

Runs a nano curriculum on an 8×8 maze in ~2 min on CPU — proves the pipeline works end-to-end before you pay for a GPU.

## Local Docker dry-run (needs NVIDIA + nvidia-docker)

```bash
make build
make run
make logs        # tail
make stop
```
