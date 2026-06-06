# deepMaze self-improvement loop

You are running inside a deepMaze RunPod container. A baseline training run has just completed under `${OUTPUT_BASE}/mlruns/`. Your job is to **iteratively improve the eval_success_rate** of the final curriculum stage by reading results, diagnosing failures, editing source, and re-running training — up to `${MAX_IMPROVE_ITERS}` iterations (default 5).

## Mission

Maximize `eval_success_rate` of the last stage in the curriculum. The baseline number is in the latest MLflow run under experiment `${MLFLOW_EXPERIMENT:-deepmaze}`. If `eval_success_rate > 0.9`, declare success and stop.

## Budget + stop conditions

- **Hard iteration cap:** `${MAX_IMPROVE_ITERS:-5}` improvement loops.
- **Hard time cap:** `${MAX_IMPROVE_HOURS:-4}` hours from container start.
- **Plateau:** stop early if 3 consecutive iterations show no improvement (Δ < 0.02).
- **Goal hit:** stop early if `eval_success_rate >= 0.9` on final stage.

## Tools available

- Read/Edit any file under `/app/` — but **only modify these directories**:
  - `agents/` — agent classes (Q, DQN, PPO, DRQN, DTQN)
  - `training/` — train_agent, evaluate_agent
  - `config/hyperparameters.py` — default hyperparams per agent type
  - `scripts/train_runpod.py` — the training entrypoint (if you need to change defaults at the script level)
- **Do not modify:** `tests/`, `web/`, `runpod/`, `notebooks/`, `infra/`, `flows/`, `Makefile`, `Dockerfile*`. Those are out of scope.
- Run training via `python scripts/train_runpod.py` with env vars you can override per iteration (set them in a shell wrapper).
- Read MLflow runs from `${OUTPUT_BASE}/mlruns/` (file store). You can use `mlflow.search_runs()` from Python, or just grep the YAML/JSON inside the run dirs.
- Use `git` — each iteration commits to branch `claude-improve`. The user reviews/cherry-picks at the end.

## Known structural issues (start here)

These are pre-existing bugs in the repo, not subtle ones. Fix the obvious one first, see if it moves the needle, before going deeper:

1. **Epsilon decay is per-step, not per-episode.** Look at `agents/drqn_agent.py::update`, `agents/dtqn_agent.py::update`, `agents/q_agent.py::update`, `agents/dqn_agent.py::update`. The `self.epsilon *= self.epsilon_decay` line fires every step, so on a 1000-step episode, ε collapses to `min_epsilon` inside the first episode. Move it to fire only on `done`, or to a new `on_episode_end()` hook called from `training/train.py::train_agent`'s outer loop.

2. **Default `buffer_capacity=200`** in `config/hyperparameters.py` is tiny — on 600-step episodes that's 1/3 of one episode. Bump to ~50000 for DRQN/DTQN defaults.

3. **`min_epsilon=0.05`** with random-action over 4 actions = 1.25% chance per step to random-walk into a bad cell even when policy is good. Could lower to 0.01.

Fix any of these in iteration 1, retrain, observe.

## The loop

```
for iter in 1..N:
    1. Read latest run from ${OUTPUT_BASE}/mlruns/. Note eval_success_rate.
    2. Diagnose: WHY is it that number? (ε crashed? loss stuck at 0? len always = max_steps?)
       Write a 3-bullet diagnosis to stdout.
    3. Propose a SINGLE change (one bug or one knob). State the hypothesis.
    4. Edit the file. Show the diff via `git diff`.
    5. Re-run: `python scripts/train_runpod.py` with the same env config as the baseline.
       (You can shrink CURRICULUM to just the final stage to speed up iterations —
        keep the last tuple of the existing CURRICULUM env var.)
    6. Read the new eval_success_rate. Δ = new - old.
    7. If Δ > 0.02: `git commit -am "iter ${iter}: <one-line>"`. Continue.
    8. If Δ <= 0.02: `git reset --hard`. Try a different hypothesis next iteration.
    9. Log to `${OUTPUT_BASE}/improve_log.tsv`: iter, hypothesis, before, after, delta, committed.
```

## What "diagnose" looks like

Examples of good 3-bullet diagnoses:

> Baseline: succ=0%, ep_len always 600, ε hit 0.05 by episode 3.
> - The replay buffer never receives a positive-reward transition (succ=0 → no terminal +1).
> - ε collapsed within episode 1 (per-step decay × 600 = 0.951 per ep is OK, but starting from initial 1.0 and decaying 0.999^600 ≈ 0.55, so by ep 5 it's near min).
> - Without exploration + without positive transitions, no Bellman signal → loss is 0 → no learning.
> Hypothesis: move decay to per-episode (fix bug #1) — predicts ε will linger near 1.0 for ~200 eps, agent finds treasure stochastically, loss becomes non-zero.

## Output discipline

- After each iteration, print a single-line summary: `[iter N] hypothesis="..." before=X.X% after=Y.Y% delta=+Z.Z% committed=yes/no`.
- At the very end, print a markdown summary of all iterations (table: iter, hypothesis, before, after, committed).
- Commit messages: `claude-improve iter N: <bug-fix-or-tuning-name>` — short, indexed.

## Out of scope (don't go here)

- Don't change the maze environment dynamics (reward shape, action space, observation shape).
- Don't add new agent types from scratch.
- Don't refactor for "cleanliness" — only changes that move the metric count.
- Don't touch tests or evaluation logic to make numbers look better. Only changes to **training** code count.
- Don't push to remote. Commit to local `claude-improve` branch only.

Begin with iteration 1. Read the baseline run first.
