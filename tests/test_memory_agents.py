"""DRQN v2 (LSTM + spatial-attention) and DTQN (transformer) both run
end-to-end on partial-view + lava mazes."""

import pytest
from maze import MazeEnvironment
from seeding import seed_everything
from train import create_agent, simulate_episode, train_agent


def _tiny_kw(agent_type):
    if agent_type == "drqn":
        return dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                    lstm_hidden=16, enc_dim=16, action_emb_dim=4)
    if agent_type == "dtqn":
        return dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                    dim=16, heads=2, layers=1, max_ctx=8)


@pytest.mark.parametrize("agent_type", ["drqn", "dtqn"])
def test_memory_agent_smoke(agent_type):
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1, n_lava=1)
    agent = create_agent(agent_type, env, **_tiny_kw(agent_type))
    train_agent(env, agent, num_episodes=3, max_steps=12)
    states, positions, _, _ = simulate_episode(env, agent, max_steps=12,
                                               at_start=True)
    assert len(states) >= 2


@pytest.mark.parametrize("agent_type", ["drqn", "dtqn"])
def test_memory_resets_on_episode_start(agent_type):
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1)
    agent = create_agent(agent_type, env, **_tiny_kw(agent_type))
    train_agent(env, agent, num_episodes=2, max_steps=8)
    agent.move(env.reset(at_start=True))
    if agent_type == "drqn":
        assert agent._hidden is not None
    else:
        assert len(agent._ctx_obs) > 0
    agent.on_episode_start()
    if agent_type == "drqn":
        assert agent._hidden is None
    else:
        assert len(agent._ctx_obs) == 0


def test_drqn_prev_action_tracked():
    """The LSTM's prev-action input should reflect the last move()."""
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1)
    agent = create_agent("drqn", env, **_tiny_kw("drqn"))
    agent.on_episode_start()
    s = env.reset(at_start=True)
    a = agent.move(s)
    assert agent._last_action == a
    agent.on_episode_start()
    assert agent._last_action == -1
