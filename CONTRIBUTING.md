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

## MLOps changes

When touching anything under `notebooks/`, `flows/`, `infra/`, `Dockerfile.prod`, `docker/`, or `.github/workflows/`:

1. Cross-check the env-var matrix in [`docs/architecture.md`](docs/architecture.md) — if you add or rename a var, update the matrix in the same commit.
2. If the change affects deploy semantics, update [`docs/deployment-guide.md`](docs/deployment-guide.md).
3. Run `python -m pytest tests/ -q` and `ruff check .` — both must stay green.
4. Notebook changes: re-validate with `python -c "import json; json.load(open('notebooks/train_agent.ipynb'))"`. The notebook is ruff-excluded — keep cells small and avoid notebook-only idioms in importable code.
