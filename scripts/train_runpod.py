"""Standalone training entrypoint for RunPod (or any Docker/CLI host).

Mirrors the notebook's `run_curriculum()` and `train_one()` so you get the
same MLflow logging + bundle output, just without Colab/Drive. Config is
read from environment variables (defaults match the notebook's "real run"
values for a 30×60 final stage). Outputs land under ${OUTPUT_BASE}.

Run locally:
    OUTPUT_BASE=/tmp/dm python scripts/train_runpod.py

On RunPod (via the entrypoint.sh from runpod/Dockerfile):
    docker run --gpus all -v dm-vol:/workspace garassinoj/deepmaze-train

Env vars (all optional — defaults shown):
    AGENTS_TO_RUN=drqn         # comma-separated
    RUN_TAG=v1
    SEED=0
    # Curriculum: semicolon-separated "W,H,n_treasures,episodes,max_steps[,seq_len]"
    CURRICULUM=10,20,2,800,180,8;20,40,4,1500,720,16;30,60,6,2500,1620,32
    # Single-stage override (skips curriculum):
    MAZE_WIDTH= MAZE_HEIGHT= N_TREASURES= NUM_EPISODES= MAX_STEPS=
    PARTIAL=5 N_LAVA=2 COLLECT_ALL=true DENSITY=0.2
    GENERATOR=dfs,random   # comma list = per-build sample (generalization)
    REGENERATE_EVERY=1 EVAL_REGENERATE=true EVAL_EPISODES=50
    RANDOM_START=true BUMP_PENALTY=-0.01
    AUX_FEATURES=false REWARD_SHAPING=true   # memory-first; aux = ablation knob
    EVAL_EVERY=0 ADVANCE_THRESHOLD=0 STAGE_MAX_REPEATS=1
    NANO=false      # true = tiny nets + learn_every=4 + 3-episode evals
                    # (CPU smoke-test; `make local` sets it)
    # EVAL_EVERY 0 = num_episodes//10. ADVANCE_THRESHOLD gates curriculum
    # promotion on periodic-eval success rate (0 disables the gate).
    EXPLORATION_DECAY=0 BUFFER_CAPACITY=0   # 0 = repo default; decay is
                                            # per-EPISODE, capacity in EPISODES
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `from train import ...` regardless of cwd.
HERE = Path(__file__).resolve().parent.parent
for sub in ("agents", "environment", "training", "utils", "config"):
    p = str(HERE / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import mlflow
import torch
from session import train_session

# ─────────────────────────── config from env ────────────────────────────

def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "y", "on")
    return cast(raw)


OUTPUT_BASE   = Path(_env("OUTPUT_BASE", "/tmp/deepmaze_runpod"))
MLRUNS_DIR    = OUTPUT_BASE / "mlruns"
ASSETS_DIR    = OUTPUT_BASE / "assets"
SHOWCASE_BASE = OUTPUT_BASE / "showcase"

AGENTS_TO_RUN     = _env("AGENTS_TO_RUN", "drqn,dtqn")
RUN_TAG           = _env("RUN_TAG", "v1")
MLFLOW_EXPERIMENT = _env("MLFLOW_EXPERIMENT", "deepmaze")
SEED              = _env("SEED", 0, int)

# Curriculum or single-stage. If MAZE_WIDTH is set, single-stage wins.
SINGLE_W = _env("MAZE_WIDTH", 0, int)
if SINGLE_W:
    STAGES = [(
        SINGLE_W,
        _env("MAZE_HEIGHT", 60, int),
        _env("N_TREASURES", 10, int),
        _env("NUM_EPISODES", 3000, int),
        _env("MAX_STEPS", 600, int),
        _env("SEQ_LEN", 0, int),
    )]
else:
    # Stage fields: W,H,n_treasures,episodes,max_steps[,seq_len].
    # seq_len scales the memory window with maze size (0 = repo default);
    # burn_in follows as seq_len//2. Weight transfer is unaffected — seq_len
    # is not shape-bearing (the LSTM unrolls any length).
    # Collect-all tours: max_steps = 3*(w+h)*n_treasures, treasure counts
    # ramp 2/4/6 so the keep-going-after-pickup behavior is learned early.
    raw = _env("CURRICULUM",
               "10,20,2,800,180,8;20,40,4,1500,720,16;30,60,6,2500,1620,32")
    STAGES = []
    for stage in raw.split(";"):
        parts = [int(x.strip()) for x in stage.split(",")]
        w, h, nt, ne, mx = parts[:5]
        sq = parts[5] if len(parts) > 5 else 0
        STAGES.append((w, h, nt, ne, mx, sq))

GENERATOR        = _env("GENERATOR", "dfs,random")  # comma list = sampled per maze build
DENSITY          = _env("DENSITY", 0.2, float)
N_LAVA           = _env("N_LAVA", 2, int)
COLLECT_ALL      = _env("COLLECT_ALL", True, bool)
PARTIAL          = _env("PARTIAL", 5, int)
REGENERATE_EVERY = _env("REGENERATE_EVERY", 1, int)
EVAL_REGENERATE  = _env("EVAL_REGENERATE", True, bool)
EVAL_EPISODES    = _env("EVAL_EPISODES", 50, int)
RANDOM_START     = _env("RANDOM_START", True, bool)
BUMP_PENALTY     = _env("BUMP_PENALTY", -0.01, float)
# Memory-first: the agent must navigate from its recurrent memory, so aux
# observation features default OFF (flip on for an ablation A/B). Shaping
# stays on — it densifies the training signal without leaking anything
# into the observation the policy sees at inference.
AUX_FEATURES     = _env("AUX_FEATURES", False, bool)
REWARD_SHAPING   = _env("REWARD_SHAPING", True, bool)
EVAL_EVERY        = _env("EVAL_EVERY", 0, int)       # 0 = num_episodes//10
ADVANCE_THRESHOLD = _env("ADVANCE_THRESHOLD", 0.0, float)  # 0 = gate off
STAGE_MAX_REPEATS = _env("STAGE_MAX_REPEATS", 1, int)

# NANO=true → tiny architectures + sparse gradient steps + small evals so a
# full pipeline cycle finishes in minutes on an old CPU. Smoke-test only —
# verifies the pipeline, NOT convergence.
NANO = _env("NANO", False, bool)
_NANO_ARCH = {
    "drqn": dict(enc_dim=16, lstm_hidden=32, action_emb_dim=4,
                 batch_size=4, seq_len=4, burn_in=1, learn_every=4),
    "dtqn": dict(dim=32, heads=2, layers=1, max_ctx=16,
                 batch_size=4, seq_len=4, burn_in=1, learn_every=4),
    "dqn":  dict(batch_size=16),
}

# 0/0.0 = use repo defaults. Decay is per EPISODE (default 0.995); the old
# 0.999995 per-step compensation constant is gone — agents no longer decay
# inside update(). BUFFER_CAPACITY is in EPISODES for drqn/dtqn.
EXPLORATION_DECAY = _env("EXPLORATION_DECAY", 0.0, float)
BUFFER_CAPACITY   = _env("BUFFER_CAPACITY", 0, int)

PRINT_EVERY     = _env("PRINT_EVERY", 50, int)
SHOWCASE_EVERY  = _env("SHOWCASE_EVERY", 500, int)
WINDOW          = _env("WINDOW", 100, int)
SHOWCASE_SPRITE = _env("SHOWCASE_SPRITE", 12, int)
SHOWCASE_FRAMES = _env("SHOWCASE_FRAMES", 300, int)


# ─────────────────────────── helpers ────────────────────────────────────

def _hr(c: str = "─", w: int = 78) -> str:
    return c * w


def _agent_overrides(agent_type: str) -> dict:
    cand = {"exploration_decay": EXPLORATION_DECAY}
    if agent_type in ("drqn", "dtqn", "dqn"):
        cand["buffer_capacity"] = BUFFER_CAPACITY
    out = {k: v for k, v in cand.items() if v}
    if NANO:
        out.update(_NANO_ARCH.get(agent_type, {}))
    return out


def train_stage(agent_type: str, run_name: str,
                width: int, height: int, n_treasures: int,
                num_episodes: int, max_steps: int,
                seq_len: int = 0,
                warm_start_path: str | None = None) -> dict:
    overrides = _agent_overrides(agent_type)
    if seq_len and agent_type in ("drqn", "dtqn"):
        overrides["seq_len"] = seq_len
        overrides["burn_in"] = max(2, seq_len // 2)
    return train_session(
        agent_type=agent_type, run_name=run_name,
        env_kw=dict(
            width=width, height=height, density=DENSITY,
            generator=GENERATOR, n_lava=N_LAVA, n_treasures=n_treasures,
            collect_all=COLLECT_ALL, partial_view=PARTIAL, seed=SEED,
            bump_penalty=BUMP_PENALTY,
            aux_features=AUX_FEATURES, reward_shaping=REWARD_SHAPING,
        ),
        num_episodes=num_episodes, max_steps=max_steps,
        assets_dir=ASSETS_DIR, showcase_dir=SHOWCASE_BASE / run_name,
        agent_overrides=overrides, warm_start_path=warm_start_path,
        random_start=RANDOM_START,
        regenerate_every=(REGENERATE_EVERY or None),
        eval_episodes=(5 if NANO else EVAL_EPISODES),
        eval_regenerate=EVAL_REGENERATE,
        eval_every=EVAL_EVERY,
        periodic_eval_episodes=(3 if NANO else 10),
        seed=SEED, window=WINDOW, print_every=PRINT_EVERY,
        showcase_every=SHOWCASE_EVERY, showcase_sprite=SHOWCASE_SPRITE,
        showcase_frames=SHOWCASE_FRAMES,
        mode_label="runpod",
    )


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    SHOWCASE_BASE.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file://{MLRUNS_DIR.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    agents = [a.strip() for a in AGENTS_TO_RUN.split(",") if a.strip()]
    print(_hr("█"))
    print("  deepMaze RunPod training")
    print(f"  output     : {OUTPUT_BASE}")
    print(f"  mlflow     : {tracking_uri}")
    print(f"  agents     : {agents}")
    print(f"  stages     : {STAGES}")
    print(f"  cuda avail : {torch.cuda.is_available()}  "
          f"device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(_hr("█"))

    all_results = {}
    for agent_type in agents:
        warm = None
        stage_results = []
        for i, (w, h, nt, ne, mx, sq) in enumerate(STAGES):
            res = None
            for attempt in range(1 + max(0, STAGE_MAX_REPEATS - 1)):
                run_name = f"{agent_type}_{RUN_TAG}_s{i}_{w}x{h}" + (
                    f"_r{attempt}" if attempt else "")
                res = train_stage(agent_type, run_name, w, h, nt, ne, mx,
                                  seq_len=sq, warm_start_path=warm)
                stage_results.append(res)
                gate = max(res["eval_success_rate"], res["best_success_rate"])
                if not ADVANCE_THRESHOLD or gate >= ADVANCE_THRESHOLD:
                    break
                # retry the same stage from its best snapshot
                warm = res["best_model_path"] or res["model_path"]
                print(f"  ⟳ stage {i} below gate "
                      f"({gate:.1%} < {ADVANCE_THRESHOLD:.1%}) — retrying")
            gate = max(res["eval_success_rate"], res["best_success_rate"])
            if ADVANCE_THRESHOLD and gate < ADVANCE_THRESHOLD:
                print(f"  ✗ stage {i} never passed the gate — stopping "
                      f"curriculum for {agent_type}")
                break
            warm = res["best_model_path"] or res["model_path"]
        all_results[agent_type] = stage_results

    print()
    print(_hr("="))
    print("  SUMMARY")
    print(_hr("="))
    for agent_type, stages in all_results.items():
        for r in stages:
            print(f"  {r['agent_type']:5s}  {r['run_name']:40s}  succ={r['eval_success_rate']:.2%}")


if __name__ == "__main__":
    main()
