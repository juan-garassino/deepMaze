# notebooks/

Colab notebooks that train heavy agents (DRQN, DTQN) — the local pytest suite stays nano-scale on 5×5 tabular Q.

| Notebook | What it does |
|---|---|
| [`train_agent.ipynb`](train_agent.ipynb) | Mounts Google Drive, clones the repo at HEAD, trains **DRQN then DTQN** sequentially on a 12×12 multi-treasure + lava maze, persists MLflow runs and `assets/<name>/` bundles to Drive. No GCP required. |

## Open in Colab

Use Colab's GitHub import: **File → Open notebook → GitHub tab → enter `juan-garassino/deepMaze` → select `notebooks/train_agent.ipynb`**.

Or direct URL pattern:
```
https://colab.research.google.com/github/juan-garassino/deepMaze/blob/main/notebooks/train_agent.ipynb
```

## Open in VS Code (with Colab compute)

The official **Google Colab** VS Code extension lets you edit the `.ipynb` locally while the kernel runs on a Colab GPU runtime:

1. Install the **Google Colab** extension (publisher: Google).
2. Open `notebooks/train_agent.ipynb` in VS Code.
3. Kernel picker → **Google Colab → connect** (sign in with the same Google account that has Colab access).
4. The `/content/deepMaze` path used in cell 2 is still valid because the runtime is still Colab.

Local Jupyter kernel (no Colab compute, CPU only) also works for the small Q-learning test case, but DRQN/DTQN training without a GPU is impractical. If going local, change `/content/deepMaze` in cell 2 to your existing checkout path.

## What the notebook needs

| Input | Where |
|---|---|
| Google Drive auth | Cell 1 prompts via `drive.mount(...)`; accept once per runtime |
| `REPO_URL` / `REPO_BRANCH` | form fields in cell 2; defaults to `github.com/juan-garassino/deepMaze.git@main` |
| `DRIVE_BASE` | form field; defaults to `/content/drive/MyDrive/deepMaze`. `mlruns/` + `assets/` get created under it |
| `AGENTS_TO_RUN` | comma-separated; default `"drqn,dtqn"` trains both in sequence |
| GPU runtime | Runtime → Change runtime type → GPU. T4 is enough for DRQN; A100 helps DTQN with batch > 32 |

## Output checklist

After a successful run you should see, in Drive at `${DRIVE_BASE}/`:
- `mlruns/` — full file-store MLflow experiment with both runs, params, per-episode metrics, eval metrics, logged artifacts
- `assets/drqn_v1/` and `assets/dtqn_v1/` — each contains `config.json` + `model.pt` + `viz/replay.webp`
- `eval_success_rate > 0` on both runs is "it works"

## Pulling bundles down + browsing MLflow locally

```bash
# Copy a bundle to use with the local backend's pretrained dropdown
rsync -a "<drive>/deepMaze/assets/drqn_v1/" assets/drqn_v1/
python web/server.py --port 8000

# Browse the MLflow runs
rsync -a "<drive>/deepMaze/mlruns/" mlruns/
mlflow ui --backend-store-uri "file://$(pwd)/mlruns"
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `drive.mount` prompts for code every cell | runtime hot-restarted | re-run cell 1 once per fresh runtime |
| Slow `mlflow.log_metrics` calls | Drive FUSE round-trips | expected; first DRQN run with 3000 episodes ≈ 20 min on T4 |
| `OSError: [Errno 28] No space left` | Colab's `/tmp` filled by replay frames | lower `MAX_STEPS` or `NUM_EPISODES`, or restart the runtime |
| Notebook hangs early in cell 6 | DRQN replay buffer warming up — first 1k steps are slow | wait; or interrupt and restart with smaller `NUM_EPISODES` |
| DTQN OOM on T4 | transformer attention with `MAX_STEPS=300` | drop `MAX_STEPS` to 200, or switch runtime to A100 |

## Promote a trained model

After the notebook finishes, the bundles live at `${DRIVE_BASE}/assets/<run_name>/`. To use them with the deployed backend:

- **Manual:** download the bundle from Drive, place at `assets/<run_name>/`, `git add` + `git push` — the deploy workflow picks it up.
- **Automated (Cloud MLflow):** if you also have a Cloud Run MLflow server, log the run there and call `python flows/promote_flow.py <mlflow-run-id>` to open a PR. See [`../flows/README.md`](../flows/README.md).
