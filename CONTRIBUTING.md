# Contributing

## Setup

```bash
pip install -r requirements-dev.txt
```

## Test

```bash
python -m pytest tests/ -q
```

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
