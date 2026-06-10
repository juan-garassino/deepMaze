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
    net: str = "mlp"


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
    net: str = "mlp"


@dataclass(frozen=True)
class DRQNHP:
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    min_epsilon: float = 0.05
    batch_size: int = 8
    seq_len: int = 8
    burn_in: int = 4
    target_sync: int = 100
    learn_every: int = 1  # gradient step every N env steps
    buffer_capacity: int = 200
    lstm_hidden: int = 128
    enc_dim: int = 64        # GridAttnEncoder output dim
    action_emb_dim: int = 16


@dataclass(frozen=True)
class DTQNHP:
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    min_epsilon: float = 0.05
    batch_size: int = 8
    seq_len: int = 16
    burn_in: int = 4
    target_sync: int = 100
    learn_every: int = 1  # gradient step every N env steps
    buffer_capacity: int = 200
    dim: int = 128
    heads: int = 4
    layers: int = 2
    max_ctx: int = 64


DEFAULTS = {"q": QHP(), "dqn": DQNHP(), "ppo": PPOHP(),
            "drqn": DRQNHP(), "dtqn": DTQNHP()}


def defaults_for(agent_type: str) -> dict:
    return asdict(DEFAULTS[agent_type])
