# notebooks/

Colab notebooks that train heavy agents (DRQN, DTQN) — the local pytest suite stays nano-scale on 5×5 tabular Q.

| Notebook | What it does |
|---|---|
| [`train_agent.ipynb`](train_agent.ipynb) | Clones the repo at HEAD, trains a DRQN or DTQN agent, logs to MLflow, emits an `assets/<name>/` bundle ready for the existing pretrained-inference path. |

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
| `MLFLOW_TRACKING_URI` | the deployed MLflow server URL (see [`infra/mlflow/README.md`](../infra/mlflow/README.md)) |
| `GOOGLE_APPLICATION_CREDENTIALS` | only required if `ASSETS_BUCKET` is set; upload the JSON via the Colab file pane and reference the path |
| `ASSETS_BUCKET` | optional — if set, the bundle is pushed directly to `gs://${ASSETS_BUCKET}/<run_name>/` and the deployed backend picks it up on next sync |
| GPU runtime | Runtime → Change runtime type → GPU. T4 is enough for DRQN; A100 helps DTQN with batch > 32 |

## Output checklist

After a successful run you should see:
- MLflow run under experiment `deepmaze` with `eval_success_rate` metric > 0
- `assets/<RUN_NAME>/config.json` + `model.pt` + `viz/replay.webp` in Colab's file pane
- Either a downloadable `<RUN_NAME>.zip` (no `ASSETS_BUCKET`) or successful GCS uploads (with bucket)

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: Set MLFLOW_TRACKING_URI` | form field left blank | paste your tracking URL into the cell 1 form |
| `OSError: [Errno 28] No space left on device` | Colab's `/tmp` filled by replay frames | lower `MAX_STEPS` or `NUM_EPISODES`, or restart the runtime |
| `requests.exceptions.ConnectionError` to MLflow | tracking server unreachable from Colab | confirm the URL is HTTPS, public, and not behind IAP yet |
| GCS `403 PERMISSION_DENIED` on upload | key lacks `roles/storage.objectAdmin` on the bucket | re-grant via `gsutil iam ch` or use a different SA |
| Notebook hangs in cell 4 | DRQN replay buffer warming up — first 1k steps are slow | wait; or interrupt and restart with smaller `NUM_EPISODES` |

## Security note

**Never paste your `GOOGLE_APPLICATION_CREDENTIALS` JSON into a notebook cell.** Even with cleared outputs, the JSON persists in the `.ipynb` and ends up in version control. Upload the file via the Colab file pane (left sidebar) and reference it by path.

## Promote a trained model

After the notebook finishes, either:

- **Manual:** download the zip, unzip into `assets/<name>/`, `git add` + `git push` — the deploy workflow picks it up.
- **Automated:** run `python flows/promote_flow.py <mlflow-run-id>` from your machine; Prefect downloads the bundle, validates it, and opens a PR. See [`../flows/README.md`](../flows/README.md).
