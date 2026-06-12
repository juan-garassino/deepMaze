"""C1 audit fixes: random_start actually randomizes, bump penalty is
configurable, eval success is terminal-based, max_steps scales with maze."""

from __future__ import annotations

from maze import EXIT, MazeEnvironment
from train import default_max_steps, evaluate_agent, run_episode


def _env(**kw):
    base = dict(width=6, height=6, generator="open", seed=0)
    base.update(kw)
    return MazeEnvironment(**base)


def test_reset_at_start_pins_agent_to_start():
    env = _env()
    for _ in range(10):
        env.reset(at_start=True)
        assert env.agent_positions[0] == env.start_pos


def test_reset_random_start_varies():
    env = _env()
    positions = {env.reset(at_start=False) is not None and env.agent_positions[0]
                 for _ in range(20)}
    assert len(positions) > 1, "random reset always landed on the same cell"


def test_bump_penalty_configurable():
    env = _env(bump_penalty=-0.01)
    env.reset(at_start=True)
    # Start is (1,1); moving up (action 0) targets the border wall → bump.
    _, reward, done, _ = env.step(0)
    assert reward == -0.01
    assert not done
    assert env.agent_positions[0] == env.start_pos


class _ScriptedAgent:
    """Replays a fixed action list, then holds the last action."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def move(self, state):
        a = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return a

    def set_deterministic(self, flag):
        pass


def test_eval_success_is_terminal_based():
    # collect_all with 2 treasures: an agent that grabs one treasure and then
    # oscillates has total_reward > 0 but never finishes → success must be 0.
    env = _env(width=7, height=7, n_treasures=2, collect_all=True)
    env.reset(at_start=True)
    treasures = sorted(env.treasure_positions)
    # Walk from (1,1) to the nearest treasure along an open maze, then loop.
    tr = treasures[0]
    path = []
    r, c = env.start_pos
    while r < tr[0]:
        path.append(2)
        r += 1
    while c < tr[1]:
        path.append(1)
        c += 1
    path += [0, 2] * 50  # oscillate forever after the first pickup
    agent = _ScriptedAgent(path)
    result = run_episode(env, agent, max_steps=40, at_start=True)
    assert result.total_reward > 0
    assert not result.done
    assert not result.success

    env2 = _env(width=7, height=7, n_treasures=2, collect_all=True)
    agent2 = _ScriptedAgent(path)
    _, _, succ = evaluate_agent(env2, agent2, num_episodes=1, max_steps=40,
                                deterministic=False)
    assert succ == 0.0


def test_run_episode_success_on_single_treasure():
    env = _env()
    env.reset(at_start=True)
    tr = env.treasure_positions[0]
    assert env.maze[tr] == EXIT
    path = [2] * (tr[0] - 1) + [1] * (tr[1] - 1)
    result = run_episode(env, _ScriptedAgent(path), max_steps=30, at_start=True)
    assert result.done and result.success


def test_default_max_steps_formula():
    env = _env(width=10, height=20)
    assert default_max_steps(env) == 3 * 30
    env_ca = _env(width=10, height=20, n_treasures=3, collect_all=True)
    assert default_max_steps(env_ca) == 3 * 30 * 3
    # non-collect_all ignores treasure count
    env_nt = _env(width=10, height=20, n_treasures=3)
    assert default_max_steps(env_nt) == 3 * 30


def test_revisit_rate_measures_oscillation():
    from train import run_episode
    env = _env()
    # oscillates between two cells forever → revisit approaches 1
    osc = run_episode(env, _ScriptedAgent([2, 0] * 20), max_steps=20,
                      at_start=True)
    assert osc.revisit_rate > 0.8
    # straight walk to the corner treasure → no revisits
    env2 = _env()
    env2.reset(at_start=True)
    tr = env2.treasure_positions[0]
    path = [2] * (tr[0] - 1) + [1] * (tr[1] - 1)
    straight = run_episode(env2, _ScriptedAgent(path), max_steps=30,
                           at_start=True)
    assert straight.success
    assert straight.revisit_rate == 0.0


def test_evaluate_agent_reports_revisit_metric():
    out: dict = {}
    env = _env()
    evaluate_agent(env, _ScriptedAgent([2, 0] * 30), num_episodes=2,
                   max_steps=10, deterministic=False, metrics_out=out)
    assert 0.0 <= out["revisit_rate"] <= 1.0
    assert out["revisit_rate"] > 0.5


def test_generator_mix_samples_both():
    used = set()
    env = MazeEnvironment(width=8, height=8, generator="dfs,open", seed=0)
    used.add(env.generator_used)
    for _ in range(12):
        env.regenerate()
        used.add(env.generator_used)
    assert used == {"dfs", "open"}
    assert env.is_solvable()


def test_generator_mix_rejects_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown generator"):
        MazeEnvironment(width=8, height=8, generator="dfs,bogus", seed=0)
