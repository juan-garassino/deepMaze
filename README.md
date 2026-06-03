# deepMaze

Maze reinforcement-learning playground. Five agents (Q / DQN / PPO / DRQN /
DTQN), partial-observation + lava + multi-treasure environments, sprite
replay (WebP/GIF/MP4), training-curve / policy / visitation / rollout PNGs,
self-contained HTML run reports, and a browser viewer that streams live
training **or** plays back pretrained models — all over the same SSE pipeline.

| | |
|---|---|
| ![training curves](docs/screenshots/curves.png) | ![policy heatmap](docs/screenshots/policy.png) |
| ![behavioral rollout](docs/screenshots/rollout.png) | ![replay](docs/screenshots/replay.webp) |

*Sample artifacts from a 7×7 Q-agent run with two treasures (`--n_treasures 2 --generator dfs --num_episodes 80`).*

## Quickstart — Docker

```bash
docker compose up --build
```

Open `http://localhost:8080`. Backend on `:8000`, frontend on `:8080`.
Mount `./assets/` to ship pretrained models trained externally (Colab).
Mount `./maze_rl_runs/` to persist training output.

### First-run sanity check

The Dockerfiles are written but the first `docker compose up --build` is
your integration test. Expected:

1. `[+] Building` for both `backend` and `frontend` images (~2–5 min, torch
   CPU wheel is the slow part).
2. `deepmaze-backend` becomes `healthy` after the `/api/health` probe
   succeeds (≤ 30 s).
3. `deepmaze-frontend` starts and nginx serves `:8080`.
4. Browse to `http://localhost:8080` — the maze editor loads. The "regenerate"
   button hits `:8000/api/maze/generate` (visible in browser dev-tools
   network tab).
5. Click `Train` to fire a tiny Q-agent → SSE stream → live canvas + charts.

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

## MLOps + GCP

| Surface | Where |
|---|---|
| Architecture (one-page wiring + data flow) | [`docs/architecture.md`](docs/architecture.md) |
| Cold-start deployment runbook | [`docs/deployment-guide.md`](docs/deployment-guide.md) |
| Colab training notebook | [`notebooks/`](notebooks/) — see [`notebooks/README.md`](notebooks/README.md) |
| Prefect flows (retrain · promote · smoke-test) | [`flows/`](flows/) — see [`flows/README.md`](flows/README.md) |
| Infra provisioning (MLflow · Cloud Run · Prefect) | [`infra/`](infra/) — see [`infra/README.md`](infra/README.md) |
| CI/CD with Slack | [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) |
| Observability (MLflow · Cloud Monitoring · Cloud Trace) | [`docs/observability.md`](docs/observability.md) |
| Original design spec | [`docs/superpowers/specs/2026-06-03-deepmaze-mlops-design.md`](docs/superpowers/specs/2026-06-03-deepmaze-mlops-design.md) |

### Train in Colab → deploy via PR (TL;DR)

1. Open `notebooks/train_agent.ipynb` in Colab, set `MLFLOW_TRACKING_URI`, run all cells.
2. The notebook logs to MLflow and emits `assets/<run_name>/` (model + config + replay).
3. Locally: `python flows/promote_flow.py <mlflow-run-id>` — Prefect downloads the bundle, validates it, commits it to `assets/`, opens a PR.
4. Merging the PR triggers `.github/workflows/deploy.yml` → Cloud Run revision with the new asset, Slack notifications on start + end.

Full step-by-step (from an empty GCP project): [`docs/deployment-guide.md`](docs/deployment-guide.md).
