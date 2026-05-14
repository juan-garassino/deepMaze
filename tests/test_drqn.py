"""DRQN trains on partial-view maze without errors."""


from maze import MazeEnvironment
from seeding import seed_everything
from train import create_agent, simulate_episode, train_agent


def test_drqn_partial_view_smoke():
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1, n_lava=1)
    agent = create_agent("drqn", env, batch_size=2, seq_len=4, burn_in=2,
                         buffer_capacity=8, lstm_hidden=16)
    train_agent(env, agent, num_episodes=3, max_steps=15)
    states, positions, _, _ = simulate_episode(env, agent, max_steps=15,
                                               at_start=True)
    assert len(states) >= 2


def test_drqn_hidden_resets_on_episode_start():
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1)
    agent = create_agent("drqn", env, batch_size=2, seq_len=4, burn_in=2,
                         buffer_capacity=8, lstm_hidden=16)
    train_agent(env, agent, num_episodes=2, max_steps=8)
    agent.move(env.reset(at_start=True))
    assert agent._hidden is not None
    agent.on_episode_start()
    assert agent._hidden is None
