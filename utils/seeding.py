"""Single entry point for seeding all stochastic libraries.

Without this, --seed only governed the environment's local RNG; torch/numpy
globals stayed unseeded and DQN/PPO runs diverged across reruns.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: Optional[int]) -> Optional[int]:
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
        # determinism over speed; cheap for tiny nets.
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
    return seed
