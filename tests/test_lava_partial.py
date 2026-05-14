
from maze import AGENT_BASE, HOLE, LAVA, MazeEnvironment


def test_lava_placed_off_shortest_path():
    env = MazeEnvironment(8, 8, density=0.0, seed=0, generator="open",
                          n_lava=4)
    assert (env.maze == LAVA).sum() == 4
    assert env.is_solvable()  # lava placed only off the path


def test_step_into_lava_terminates_with_penalty():
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open",
                          n_lava=0)
    env.maze[2, 2] = LAVA
    env.agent_positions = [(2, 1)]
    obs, r, done, _ = env.step(1)  # right -> (2,2) lava
    assert done
    assert r == -1.0


def test_lava_blocks_spawn():
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          n_lava=5)
    # 100 reset draws should never spawn on lava
    for _ in range(100):
        env.reset()
        for p in env.agent_positions:
            assert env.maze[p] != LAVA


def test_partial_view_shape_and_centered():
    env = MazeEnvironment(7, 7, seed=0, generator="open", partial_view=2)
    env.reset(at_start=True)
    obs = env.get_observation()
    assert obs.shape == (5, 5)
    assert obs[2, 2] >= AGENT_BASE  # agent at center


def test_partial_view_pads_with_hole_at_borders():
    env = MazeEnvironment(7, 7, seed=0, generator="open", partial_view=3)
    env.reset(at_start=True)
    obs = env.get_observation()
    assert obs.shape == (7, 7)
    # at start=(1,1) the top-left of the window is at (-2,-2) -> HOLE pad
    assert obs[0, 0] == HOLE
    assert obs[1, 0] == HOLE
