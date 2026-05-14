# Pretrained models

Drop a directory here matching the layout:

```
assets/<model_name>/
    config.json    # mirrors the one written by MazeManager.save_config
    model.pt       # state_dict for DQN/PPO/DRQN/DTQN
    model.pkl      # tabular Q (alternative to model.pt)
    viz/replay.webp  # optional preview, surfaces in the runs grid
```

`config.json` must include at least `agent_type`, plus the env params it
was trained on (`maze_width`, `maze_height`, `generator`, `density`,
`n_lava`, `n_treasures`, `partial`, etc.). The web `/api/models` endpoint
auto-discovers any subdirectory containing both files.

Heavy models (CNN/LSTM/Transformer) are trained externally — e.g. Colab —
and dropped in here. The local `pytest` suite trains only tiny Q-agents
on 5×5 mazes to keep iteration fast.
