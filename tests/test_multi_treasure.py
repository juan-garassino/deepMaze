"""Multi-treasure env: placement, reachability, terminal behaviour."""

import numpy as np
from maze import EXIT, MazeEnvironment


def test_three_treasures_placed_and_reachable():
    env = MazeEnvironment(10, 10, density=0.0, seed=0, generator="dfs",
                          n_treasures=3)
    exits = list(zip(*np.where(env.maze == EXIT)))
    assert len(exits) == 3
    assert env.is_solvable()  # checks reachability of every treasure


def test_default_n_treasures_is_one():
    env = MazeEnvironment(8, 8, density=0.0, seed=0, generator="open")
    exits = list(zip(*np.where(env.maze == EXIT)))
    assert len(exits) == 1
    assert env.treasure_pos == env.treasure_positions[0]


def test_step_into_treasure_terminates_collect_one():
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open",
                          n_treasures=1)
    # Force a known config and step onto the goal.
    env.agent_positions = [(env.height - 2, env.width - 3)]
    _, r, done, _ = env.step(1)  # right -> EXIT
    assert done and r == 1.0


def test_collect_all_episode_ends_after_last_treasure():
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open",
                          n_treasures=2, collect_all=True)
    assert len(env.treasure_positions) == 2
    # consume first
    t0 = env.treasure_positions[0]
    env.agent_positions = [t0]
    obs, r, done, _ = env.step(0)  # no-op-ish; we already are on it
    # Actually we need to STEP INTO the cell. Reset and walk into t0:
    env.reset(at_start=True)
    # Move agent next to t0 and step into it manually.
    env.agent_positions = [(t0[0] - 1, t0[1])] if t0[0] > 1 else [(t0[0], t0[1] - 1)]
    action = 2 if t0[0] > 1 else 1
    _, r1, done1, _ = env.step(action)
    assert r1 == 1.0
    assert done1 is False  # one still left

    t1 = env.treasure_positions[1]
    env.agent_positions = [(t1[0] - 1, t1[1])] if t1[0] > 1 else [(t1[0], t1[1] - 1)]
    action = 2 if t1[0] > 1 else 1
    _, r2, done2, _ = env.step(action)
    assert r2 == 1.0
    assert done2 is True


def test_reset_restores_collected_treasures():
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open",
                          n_treasures=2, collect_all=True)
    t0 = env.treasure_positions[0]
    env.maze[t0] = 1  # simulate consumption
    env.reset(at_start=True)
    assert env.maze[t0] == EXIT
