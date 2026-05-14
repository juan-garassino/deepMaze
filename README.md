# deepMaze

Maze reinforcement-learning playground. Five agents (Q / DQN / PPO / DRQN /
DTQN), partial-observation + lava + multi-treasure environments, sprite
replay (WebP/GIF/MP4), training-curve / policy / visitation / rollout PNGs,
self-contained HTML run reports, and a browser viewer that streams live
training **or** plays back pretrained models — all over the same SSE pipeline.

## Quickstart — Docker

```bash
docker compose up --build
```

Open `http://localhost:8080`. Backend on `:8000`, frontend on `:8080`.
Mount `./assets/` to ship pretrained models trained externally (Colab).
Mount `./maze_rl_runs/` to persist training output.

## Quickstart — local

```bash
pip install -r requirements.txt

# Train a tiny Q-agent on a 6x6 multi-treasure maze
python main.py --agent_type q --n_treasures 3 --generator dfs \
  --maze_width 6 --maze_height 6 --num_episodes 100

# Standalone web viewer (Train + Pretrained inference + Runs browser)
python web/server.py --port 8000
# then http://localhost:8000
```

## Pretrained models

Drop a directory under `./assets/<name>/`:

```
assets/<name>/
    config.json    # at minimum: agent_type + env params
    model.pt       # or model.pkl for tabular Q
    viz/replay.webp  # optional
```

The UI auto-discovers it via `/api/models` and exposes a `▶ Watch` button
that streams a greedy episode through the canvas + memory strip.

## Tests

```bash
python -m pytest tests/ -q
```

Local tests train only tiny tabular Q-agents on 5×5 mazes — heavy models
(DRQN/DTQN/CNN) are exercised via inference only, with training done in
Colab. See `CLAUDE.md` for architectural seams and `CONTRIBUTING.md` for
the dev loop.
