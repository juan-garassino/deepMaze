# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A maze-based reinforcement learning playground with a full visualization stack — Q-learning / DQN / PPO agents, sprite-based replay (WebP/GIF/MP4), training-curve plots, policy + visitation heatmaps, and a FastAPI + vanilla-JS browser viewer.

Ported and modernized from two local references; **do not** edit those references, treat them as read-only history:

- `~/Code/006-research-prototypes/RL-maze-reinforcement-learning/001-maze-rl/` — modern Python implementation (structural source).
- `~/Code/001-archives/008-mini-networks-legacy/008-maze-rl/miniReinforcedMaze/` — older sprite renderer with Q-overlay (visual reference).
- External inspiration only: `github.com/awjuliani/web-rl-playground`.

## Layout

```
deepMaze/
├── agents/       base_agent / q_agent / dqn_agent / ppo_agent
├── config/       (reserved)
├── environment/  maze.py — MazeEnvironment + RenderMaze
├── training/     train.py + recorders.py
├── tests/        pytest suite
├── utils/        manager.py + viz_events.py + visualizations.py + replay_buffer.py
├── web/          FastAPI server + static/ (HTML + JS canvas + Chart.js)
└── main.py       CLI entrypoint
```

**Import convention.** `main.py`, `tests/conftest.py`, and `web/server.py` prepend each subdir to `sys.path` so files inside use bare sibling imports (e.g. `from maze import MazeEnvironment`). When adding a new top-level dir, register it in all three sys.path blocks.

## Architectural seams

**EventBus** (`utils/viz_events.py`). Single typed pub/sub channel — `StepEvent`, `EpisodeEvent`, `PolicyEvent`, `RunEvent`. Training emits; recorders consume. Adding a viz target = one `bus.subscribe(handler)`. Never plumb metrics through return values.

**Recorders** (`training/recorders.py`). `MetricsCollector`, `TrajectoryCollector`, `TqdmTail`, `ReplayRecorder`. Pure subscribers; stateless w.r.t. training.

**Agent factory** (`training/train.py::create_agent`). Dispatches on `'q' | 'dqn' | 'ppo'`. Adding an algorithm = new class in `agents/` + branch here.

**MazeManager** (`utils/manager.py`). Single owner of per-run artifact paths (`maze_rl_runs/run_TS/...`). All saves route through it — never write artifacts from agents/training directly. `viz_dir()` returns the per-run viz folder.

**RenderMaze.save(path, fmt=...)**. Default format is **WebP** (≈10× smaller than GIF). Supports `gif` and `mp4` (requires `imageio[ffmpeg]`). Use `frame_skip` / `max_frames` to bound output size; the last frame is always kept.

## Visualization surfaces

| Surface | Where | Trigger |
|---|---|---|
| Replay (WebP/GIF/MP4) | `RenderMaze` → `MazeManager.save_replay` | post-training, greedy episode |
| Training curves | `visualizations.plot_training_curves` | post-training from `MetricsCollector` |
| Policy + V(s) heatmap | `visualizations.plot_policy_heatmap` | post-training; accepts dict (tabular) or callable (NN) |
| Visitation heatmap | `visualizations.plot_visitation` | post-training from `TrajectoryCollector` |
| Live CLI tail | `recorders.TqdmTail` | `--live` or TTY |
| Web viewer | `web/server.py` SSE + `static/app.js` | `--live_web` or `python web/server.py` |

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

Heavy training (CNN/LSTM/Transformer) belongs in Colab. The local test
suite trains only tiny tabular Q-agents on 5×5 mazes.

## Docker

Two services orchestrated by `docker-compose.yml`:

| Service  | Image                  | Port | Mounts                             |
|----------|------------------------|------|------------------------------------|
| backend  | `Dockerfile`           | 8000 | `./maze_rl_runs`, `./assets`       |
| frontend | `Dockerfile.frontend`  | 8080 | (none)                             |

The frontend is plain nginx serving `web/static`. `${API_BASE_URL}` is
substituted into `web/static/config.js` at container start; the JS reads
`window.API_BASE_URL` and prefixes every `fetch()` / `EventSource`. CORS
on the backend is governed by `CORS_ORIGINS` (comma-separated, set in
compose).

For non-Docker dev `python web/server.py` still works — `config.js`
detects the unsubstituted template and falls back to same-origin.

## Run artifacts

```
maze_rl_runs/run_YYYYMMDD_HHMMSS/
    config.json   results.json   maze_rl.log   model.{pt,pkl}
    viz/
        replay.webp      curves.png      policy.png      visitation.png
```

## Known constraints

- `find_empty_cell` falls back to `start_pos` if the maze has no walkable cells (don't crank `--density` above 0.6).
- The web SSE handler subscribes a `queue.Queue`; when the queue saturates (>4096 events) the oldest is dropped to keep the live feed responsive. Saved artifacts are unaffected — they come from in-process subscribers.
- Tabular Q-learning's policy heatmap uses each cell's observation as a key; unvisited cells show `NaN`. The `rollout.png` (behavioral viz) is the right answer for those agents.
- Sprite sheet format: 16×16 source tiles; required sprite indices: `0=HOLE, 1=LAND, 2=LAVA, 3=EXIT, 4=AGENT`. Cell-value AGENT_BASE is 5 (agents are `5 + agent_index`).
- Pretrained-model `config.json` must match the architecture: shape mismatches at `load_state_dict` time will raise.
- Multi-treasure: `n_treasures > 1` places extras on reachable LAND; lava placement excludes all start→treasure paths. `collect_all=True` keeps the episode running until every treasure is consumed.

## Workspace context

`005-products/CLAUDE.md` carries the broader bootcamp conventions. deepMaze is **not** part of the bootcamp curriculum.
