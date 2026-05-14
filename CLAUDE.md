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

# Q-learning, placeholder sprites (no asset needed)
python main.py --agent_type q --maze_width 8 --maze_height 8 \
  --num_episodes 200 --seed 0

# DQN with custom sprite sheet
python main.py --agent_type dqn --image_path assets --sprite_files sprites.png

# Live CLI tail
python main.py --live --num_episodes 1000

# Live web viewer alongside training (port 8000)
python main.py --live_web --web_port 8000

# Standalone web viewer (draw your own maze, train from the browser)
python web/server.py --port 8000

# Tests
python -m pytest tests/ -q
```

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
- Tabular Q-learning's policy heatmap uses each cell's observation as a key; if exploration never visited a cell, that cell's `V` is `NaN` and gets masked.
- Sprite sheet format: 16×16 source tiles; `RenderMaze.crop_images` resizes to `--sprite_size` (default 32). Required sprite indices: `0=HOLE, 1=LAND, 2=LAVA, 3=EXIT, 4=AGENT`.

## Workspace context

`005-products/CLAUDE.md` carries the broader bootcamp conventions. deepMaze is **not** part of the bootcamp curriculum.
