"""Single entry point for seeding all stochastic libraries.

Without this, --seed only governed the environment's local RNG; torch/numpy
globals stayed unseeded and DQN/PPO runs diverged across reruns.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int | None) -> int | None:
    if seed is None:
        return None
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Strict determinism (True) requires CUBLAS_WORKSPACE_CONFIG and
        # breaks attention ops with no deterministic impl. Speed over strict
        # — manual_seed still gives bit-stable runs for our op set.
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
    return seed
