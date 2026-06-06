# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A maze-based reinforcement learning playground with a full visualization stack — five agents (Q-learning / DQN / PPO / **DRQN** / **DTQN**, the latter two memory-equipped), sprite-based replay (WebP/GIF/MP4), training-curve plots, policy + visitation + behavioral-rollout heatmaps, and a FastAPI + vanilla-JS browser viewer. Supports live training **and** pretrained-model inference over the same SSE pipeline.

Ported and modernized from two local references; **do not** edit those references, treat them as read-only history:

- `~/Code/006-research-prototypes/RL-maze-reinforcement-learning/001-maze-rl/` — modern Python implementation (structural source).
- `~/Code/001-archives/008-mini-networks-legacy/008-maze-rl/miniReinforcedMaze/` — older sprite renderer with Q-overlay (visual reference).
- External inspiration only: `github.com/awjuliani/web-rl-playground`.

## Layout

```
deepMaze/
├── agents/       base_agent / q_agent / dqn_agent / ppo_agent / drqn_agent / dtqn_agent / nets / encoders
├── config/       (reserved)
├── environment/  maze.py — MazeEnvironment + RenderMaze
├── training/     train.py + recorders.py
├── tests/        pytest suite
├── utils/        manager.py + viz_events.py + visualizations.py + replay_buffer.py
├── web/          FastAPI server + static/ + otel.py (Cloud Trace instrumentation)
├── notebooks/    train_agent.ipynb — Colab DRQN/DTQN trainer → MLflow + assets bundle
├── flows/        Prefect flows — retrain / promote / smoke-test
├── infra/        mlflow/ (Cloud Run + Cloud SQL + GCS) · cloudrun/service.yaml · prefect/
├── docker/       entrypoint.sh (dev) + entrypoint.prod.sh (GCS asset sync + gunicorn)
└── main.py       CLI entrypoint
```

**Import convention.** `main.py`, `tests/conftest.py`, and `web/server.py` prepend each subdir to `sys.path` so files inside use bare sibling imports (e.g. `from maze import MazeEnvironment`). When adding a new top-level dir, register it in all three sys.path blocks.

## Architectural seams

**EventBus** (`utils/viz_events.py`). Single typed pub/sub channel — `StepEvent`, `EpisodeEvent`, `PolicyEvent`, `RunEvent`. Training emits; recorders consume. Adding a viz target = one `bus.subscribe(handler)`. Never plumb metrics through return values.

**Recorders** (`training/recorders.py`). `MetricsCollector`, `TrajectoryCollector`, `TqdmTail`, `ReplayRecorder`. Pure subscribers; stateless w.r.t. training.

**Agent factory** (`training/train.py::create_agent`). Dispatches on `'q' | 'dqn' | 'ppo' | 'drqn' | 'dtqn'`. Adding an algorithm = new class in `agents/` + dataclass in `config/hyperparameters.py` + branch here.

**MazeManager** (`utils/manager.py`). Single owner of per-run artifact paths (`maze_rl_runs/run_TS/...`). All saves route through it — never write artifacts from agents/training directly. `viz_dir()` returns the per-run viz folder.

**RenderMaze.save(path, fmt=...)**. Default format is **WebP** (≈10× smaller than GIF). Supports `gif` and `mp4` (requires `imageio[ffmpeg]`). Use `frame_skip` / `max_frames` to bound output size; the last frame is always kept.

## Visualization surfaces

| Surface | Where | Trigger |
|---|---|---|
| Replay (WebP/GIF/MP4) | `RenderMaze` → `MazeManager.save_replay` | post-training, greedy episode |
| Training curves | `visualizations.plot_training_curves` | post-training from `MetricsCollector` |
| Policy + V(s) heatmap | `visualizations.plot_policy_heatmap` | post-training; tabular dict OR agent w/ `q_values_batch` |
| Behavioral rollout | `visualizations.plot_behavioral_rollout` | per-cell greedy rollout → arrows. The honest answer for partial-obs / memory agents |
| Visitation heatmap | `visualizations.plot_visitation` | post-training from `TrajectoryCollector` |
| HTML run report | `utils/report.py::write_html_report` | post-training; base64-inlined; shareable standalone |
| Memory strip (live) | `static/app.js drawMemory()` | per-step SSE; DRQN hidden state / DTQN attention row |
| Live CLI tail | `recorders.TqdmTail` | `--live` or TTY |
| Web viewer | `web/server.py` SSE + `static/app.js` | `--live_web`, `python web/server.py`, or `docker compose up` |

## Common commands

```bash
pip install -r requirements.txt

# Q-learning, placeholder sprites
python main.py --agent_type q --maze_width 8 --maze_height 8 \
  --num_episodes 200 --seed 0

# Multi-treasure DFS maze with lava
python main.py --agent_type q --n_treasures 3 --n_lava 2 --generator dfs

# Collect-all variant — episode only ends after ALL treasures
python main.py --n_treasures 3 --collect_all

# Live web viewer alongside training (port 8000)
python main.py --live_web --web_port 8000

# Standalone web viewer (Train + Pretrained inference + Runs browser)
python web/server.py --port 8000

# Docker (split: backend :8000, frontend :8080)
docker compose up --build
# then visit http://localhost:8080

# Tests
python -m pytest tests/ -q

# MLflow tracking server — local dev
docker compose -f infra/mlflow/docker-compose.local.yml up --build
# → http://localhost:5000

# MLflow tracking server — GCP (Cloud Run + Cloud SQL + GCS)
bash infra/mlflow/deploy.sh    # see infra/mlflow/README.md for required env

# Prefect flows (one-time pool setup; then deploy)
prefect work-pool create --type process default-process
prefect deploy --all --prefect-file flows/prefect.yaml
python flows/promote_flow.py <mlflow-run-id>
```

## Pretrained inference

Pretrained models (typically trained externally on Colab) live in
`assets/<name>/`. Layout matches a training run:

```
assets/<name>/
    config.json       # at minimum: agent_type + env params
    model.pt          # state_dict (or model.pkl for tabular Q)
    viz/replay.webp   # optional preview thumbnail
```

`GET /api/models` enumerates both `assets/*` and `maze_rl_runs/run_*` that
contain `config.json` + a model file. `POST /api/inference` loads the
chosen one, streams a greedy episode through the existing SSE pipeline.

UI flow:
- On `/` (Train page), toggle "Load pretrained" → pick model from
  dropdown → choose maze source (same / fresh / custom-painted) → ▶ Watch.
- On `/runs`, each card has a `▶ watch` shortcut that redirects to
  `/?inference=<name>` and auto-fires the watch flow.

Heavy training (CNN/LSTM/Transformer) belongs in Colab or RunPod. The local test
suite trains only tiny tabular Q-agents on 5×5 mazes.

## Training surfaces

| Where | Purpose | Entry point | Notes |
|---|---|---|---|
| **Local** | Pipeline smoke-test (~2 min CPU) | `notebooks/train_agent.ipynb` (`NANO_LOCAL=True`) or `make local` | Auto-shrinks config to 8×8 maze / 200 eps. Verifies pipeline, NOT convergence. |
| **Colab** | Real training, interactive | same notebook from Colab UI | Mounts Drive, file:// MLflow on Drive, see `notebooks/README.md`. |
| **RunPod** | Real training, scriptable, autonomous | `runpod/Dockerfile` + `scripts/train_runpod.py` | Pattern from `005-products/020-autoresearch`. `make build push run`. Optional `CLAUDE_SELF_IMPROVE=true` mode runs Claude Code with `--dangerously-skip-permissions` after training, given `runpod/program.md` as the autonomous loop spec. `make improve API_KEY=sk-...`. |

## Docker

Two services orchestrated by `docker-compose.yml` (dev):

| Service  | Image                  | Port | Mounts                             |
|----------|------------------------|------|------------------------------------|
| backend  | `Dockerfile`           | 8000 | `./maze_rl_runs`, `./assets`       |
| frontend | `Dockerfile.frontend`  | 8080 | (none)                             |

`Dockerfile.prod` builds the slim Cloud Run image: drops dev deps, adds gunicorn + OTEL + gsutil, and `docker/entrypoint.prod.sh` syncs `gs://${ASSETS_BUCKET}/` → `/app/assets/` at startup. CI builds + pushes it to Artifact Registry via `.github/workflows/deploy.yml`.

The frontend is plain nginx serving `web/static`. `${API_BASE_URL}` is
substituted into `web/static/config.js` at container start; the JS reads
`window.API_BASE_URL` and prefixes every `fetch()` / `EventSource`. CORS
on the backend is governed by `CORS_ORIGINS` (comma-separated, set in
compose).

For non-Docker dev `python web/server.py` still works — `config.js`
detects the unsubstituted template and falls back to same-origin.

> `CORS_ORIGINS=*` is the default for dev only. Set it to the explicit
> frontend origin (e.g. `https://maze.example.com`) for any real deploy.

## Run artifacts

```
maze_rl_runs/run_YYYYMMDD_HHMMSS/
    config.json   results.json   maze_rl.log
    model.{pt,pkl}   model.best.{pt,pkl}   best_eval.json
    viz/
        replay.webp   curves.png   policy.png
        visitation.png   rollout.png   report.html
```

## Known constraints

- `find_empty_cell` falls back to `start_pos` if the maze has no walkable cells (don't crank `--density` above 0.6).
- The web SSE handler subscribes a `queue.Queue`; when the queue saturates (>4096 events) the oldest is dropped to keep the live feed responsive. Saved artifacts are unaffected — they come from in-process subscribers.
- Tabular Q-learning's policy heatmap uses each cell's observation as a key; unvisited cells show `NaN`. The `rollout.png` (behavioral viz) is the right answer for those agents.
- Sprite sheet format: 16×16 source tiles; required sprite indices: `0=HOLE, 1=LAND, 2=LAVA, 3=EXIT, 4=AGENT`. Cell-value AGENT_BASE is 5 (agents are `5 + agent_index`).
- Pretrained-model `config.json` must match the architecture: shape mismatches at `load_state_dict` time will raise.
- Multi-treasure: `n_treasures > 1` places extras on reachable LAND; lava placement excludes all start→treasure paths. `collect_all=True` keeps the episode running until every treasure is consumed.

## MLOps surfaces

| Surface | What it does | Where |
|---|---|---|
| Colab notebook (A) | mounts Drive, clones repo, trains DRQN **and** DTQN in sequence, persists MLflow runs + `assets/<name>/` bundles to Drive (no GCP needed) | `notebooks/train_agent.ipynb` |
| MLflow server (B) | experiment tracking + model registry; Cloud Run + Cloud SQL + GCS | `infra/mlflow/` |
| Cloud Run backend (C) | slim prod image; GCS asset hot-sync at startup | `Dockerfile.prod` + `infra/cloudrun/service.yaml` |
| GHA + Slack (E) | OIDC → GAR build/push → Cloud Run deploy + Slack notifications | `.github/workflows/deploy.yml` |
| Prefect flows (D) | retrain (watch MLflow), promote (run_id → PR or GCS), daily smoke test | `flows/` |
| OTEL / Cloud Trace (F) | per-request spans from FastAPI | `web/otel.py`; see `docs/observability.md` |

Required env / secrets (set in GH repo secrets/vars for deploy; locally via `.env`):
`GCP_PROJECT_ID`, `GCP_REGION`, `GAR_REPO`, `CLOUD_RUN_SERVICE`, `CLOUD_RUN_SA_EMAIL`, `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`, `ASSETS_BUCKET`, `CORS_ORIGINS`, `MLFLOW_TRACKING_URI`, `SLACK_WEBHOOK_URL`, `PREFECT_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS` (local only). Auth on the public Cloud Run service + public MLflow is **out of scope** per the spec; tighten with IAP before any real deploy.

Reading order: **[`docs/architecture.md`](docs/architecture.md)** for the single-screen wiring → **[`docs/deployment-guide.md`](docs/deployment-guide.md)** for the cold-start runbook → **[`flows/README.md`](flows/README.md)** + **[`infra/README.md`](infra/README.md)** + **[`notebooks/README.md`](notebooks/README.md)** for per-area reference. Original design intent: **[`docs/superpowers/specs/2026-06-03-deepmaze-mlops-design.md`](docs/superpowers/specs/2026-06-03-deepmaze-mlops-design.md)**.

## Workspace context

`005-products/CLAUDE.md` carries the broader bootcamp conventions. deepMaze is **not** part of the bootcamp curriculum.
