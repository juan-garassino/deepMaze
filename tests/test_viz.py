
import numpy as np
import pytest
from maze import MazeEnvironment, RenderMaze
from visualizations import (
    plot_policy_heatmap,
    plot_reward_landscape,
    plot_training_curves,
    plot_visitation,
)
from viz_events import EpisodeEvent, EventBus, RunEvent, StepEvent


def _ep(i, r=1.0, l=10, eps=0.1, loss=None, success=True):
    return EpisodeEvent(episode=i, total_reward=r, length=l,
                        epsilon=eps, loss=loss, success=success)


def test_event_bus_dispatch():
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append)
    bus.publish(_ep(0))
    bus.publish(RunEvent(kind="end"))
    assert len(seen) == 2
    assert isinstance(seen[0], EpisodeEvent)


def test_event_bus_queue():
    bus = EventBus()
    q = bus.subscribe_queue(maxsize=4)
    for i in range(3):
        bus.publish(_ep(i))
    assert q.qsize() == 3


def test_event_to_json():
    s = StepEvent(episode=0, step=0, state=np.zeros((3, 3), dtype=np.uint8),
                  position=(1, 1), action=0, reward=0.0, done=False,
                  q_values=np.array([0.1, 0.2, 0.3, 0.4]))
    js = s.to_json()
    assert js["type"] == "step"
    assert js["q_values"] == [0.1, 0.2, 0.3, 0.4]
    assert isinstance(js["state"], list)


def test_plot_curves_empty(tmp_path):
    out = tmp_path / "curves.png"
    plot_training_curves([], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_curves_with_data(tmp_path):
    eps = [_ep(i, r=np.random.randn(), l=10 + i, loss=0.1 if i % 2 else None)
           for i in range(20)]
    out = tmp_path / "curves.png"
    plot_training_curves(eps, str(out))
    assert out.stat().st_size > 0


def test_plot_visitation(tmp_path):
    env = MazeEnvironment(8, 8, seed=0)
    trajs = [[env.start_pos, (1, 2), (2, 2), env.treasure_pos]]
    out = tmp_path / "v.png"
    plot_visitation(trajs, env, str(out))
    assert out.exists()


def test_plot_policy_heatmap_callable(tmp_path):
    env = MazeEnvironment(6, 6, seed=0)
    out = tmp_path / "p.png"
    plot_policy_heatmap(lambda s: np.random.rand(4), env, str(out))
    assert out.exists()


def test_plot_reward_landscape(tmp_path):
    env = MazeEnvironment(6, 6, seed=0)
    out = tmp_path / "l.png"
    plot_reward_landscape(env, str(out))
    assert out.exists()


@pytest.mark.parametrize("fmt", ["gif", "webp"])
def test_render_save_formats(tmp_path, fmt):
    env = MazeEnvironment(6, 6, seed=0)
    rm = RenderMaze(RenderMaze.placeholder_sprites(16))
    for _ in range(8):
        obs, _, done, _ = env.step(np.random.randint(0, 4))
        rm.add(obs, env.agent_positions[0])
        if done:
            break
    out = tmp_path / f"replay.{fmt}"
    rm.save(str(out), fmt=fmt, sprite_size=16)
    assert out.exists() and out.stat().st_size > 0


def test_frame_skip_keeps_last_frame():
    env = MazeEnvironment(6, 6, seed=0)
    rm = RenderMaze(RenderMaze.placeholder_sprites(16))
    for i in range(20):
        env.step(i % 4)
        rm.add(env.get_observation(), env.agent_positions[0])
    idxs = rm._frame_indices(frame_skip=5, max_frames=None)
    assert idxs[-1] == len(rm) - 1
