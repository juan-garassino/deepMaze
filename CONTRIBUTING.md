# Contributing

## Setup — local

```bash
pip install -r requirements-dev.txt
```

## Setup — Docker (split: backend :8000, frontend :8080)

```bash
docker compose up --build
```

Live source rebuild requires a manual `docker compose build`; for fast
iteration prefer the local Python loop.

## Test

```bash
python -m pytest tests/ -q
```

Local tests stay nano: tabular Q on 5×5 mazes only. Heavy agents
(CNN/LSTM/Transformer) are tested via inference only, with trained weights
loaded from `assets/` (Colab-trained).

## Lint

```bash
ruff check .
```

## Layout

See [CLAUDE.md](./CLAUDE.md) for directory conventions, the `EventBus`
seam, and visualization surfaces.

## Adding a new agent

1. Subclass `BaseAgent` in `agents/`.
2. Add a hyperparameter dataclass in `config/hyperparameters.py` and register in `DEFAULTS`.
3. Branch on the new agent_type in `training/train.py::create_agent`.
4. Add a smoke test under `tests/`.
