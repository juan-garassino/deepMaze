import pytest
from maze import HOLE, MazeEnvironment


@pytest.mark.parametrize("generator", ["random", "dfs", "open"])
@pytest.mark.parametrize("size", [6, 10, 16])
@pytest.mark.parametrize("density", [0.1, 0.3, 0.5])
def test_ensure_solvable_guarantee(generator, size, density):
    """Across 25 seeds × all (gen,size,density), every maze must be solvable."""
    for seed in range(25):
        env = MazeEnvironment(size, size, density=density, seed=seed,
                              generator=generator)
        assert env.is_solvable(), (
            f"gen={generator} size={size} density={density} seed={seed} "
            f"not solvable")


def test_unsolvable_when_flag_off():
    """Sanity: without the guarantee, high-density random mazes fail often."""
    fails = 0
    for s in range(50):
        env = MazeEnvironment(10, 10, density=0.5, seed=s,
                              generator="random", ensure_solvable=False)
        if not env.is_solvable():
            fails += 1
    assert fails > 10, f"expected many failures, got {fails}"


def test_dfs_has_walls():
    env = MazeEnvironment(11, 11, seed=0, generator="dfs")
    interior = env.maze[1:-1, 1:-1]
    wall_frac = (interior == HOLE).mean()
    assert 0.2 < wall_frac < 0.7, f"DFS wall fraction looked odd: {wall_frac}"


def test_open_has_no_interior_walls():
    env = MazeEnvironment(8, 8, generator="open")
    interior = env.maze[1:-1, 1:-1]
    assert (interior == HOLE).sum() == 0


def test_start_and_treasure_marked():
    env = MazeEnvironment(10, 10, seed=0)
    sr, sc = env.start_pos; tr, tc = env.treasure_pos
    from maze import EXIT, START
    assert env.maze[sr, sc] == START
    assert env.maze[tr, tc] == EXIT
