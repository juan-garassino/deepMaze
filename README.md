# deepMaze

Maze reinforcement-learning playground with a full visualization stack:
Q-learning / DQN / PPO agents, WebP/GIF/MP4 replay, training-curve plots,
policy + visitation heatmaps, and an in-browser canvas viewer.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Q-learning on an 8×8 maze, placeholder sprites — no asset files needed
python main.py --agent_type q --maze_width 8 --maze_height 8 \
  --num_episodes 200 --seed 0

# DQN with a custom sprite sheet
python main.py --agent_type dqn --image_path assets --sprite_files sprites.png

# Live CLI tail
python main.py --live --num_episodes 1000

# Web viewer alongside training
python main.py --live_web --web_port 8000

# Standalone web viewer (draw a maze, train from the browser)
python web/server.py --port 8000
```

Each run writes to `maze_rl_runs/run_YYYYMMDD_HHMMSS/`:

```
config.json   results.json   model.{pt,pkl}   maze_rl.log
viz/replay.webp   viz/curves.png   viz/policy.png   viz/visitation.png
```

## Test

```bash
python -m pytest tests/ -q
```

## Layout

See `CLAUDE.md` for the directory layout, architectural seams (`EventBus`, recorders, `MazeManager`), and the visualization surface inventory.
