"""Single source of truth for default agent hyperparameters.

Agent __init__ signatures still accept the same kwargs — the dataclasses
here just centralise the defaults so train.create_agent and the web server
can pull them without re-typing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QHP:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    min_epsilon: float = 0.01


@dataclass(frozen=True)
class DQNHP:
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    min_epsilon: float = 0.01
    batch_size: int = 64
    target_sync: int = 200
    buffer_capacity: int = 10000


@dataclass(frozen=True)
class PPOHP:
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    n_steps: int = 256
    epochs: int = 4
    minibatches: int = 4
    gae_lambda: float = 0.95


DEFAULTS = {"q": QHP(), "dqn": DQNHP(), "ppo": PPOHP()}


def defaults_for(agent_type: str) -> dict:
    return asdict(DEFAULTS[agent_type])
