"""Training / evaluation / simulation. Publishes events to an EventBus."""

from __future__ import annotations

import numpy as np
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent
from dtqn_agent import DTQNAgent
from maze import MazeEnvironment
from ppo_agent import PPOAgent
from q_agent import QAgent
from viz_events import EpisodeEvent, EventBus, PolicyEvent, RunEvent, StepEvent


def create_agent(agent_type: str, env: MazeEnvironment, **kwargs):
    from hyperparameters import defaults_for
    obs_shape = env.get_observation().shape  # (h, w)
    state_size = int(np.prod(obs_shape))
    action_size = env.action_size
    grid_shape = obs_shape if len(obs_shape) == 2 else None
    hp = {**defaults_for(agent_type), **{k: v for k, v in kwargs.items() if v is not None}}
    if agent_type == "q":
        return QAgent(action_size=action_size, **hp)
    if agent_type == "dqn":
        if hp.get("net") == "cnn":
            hp["grid_shape"] = grid_shape
        return DQNAgent(state_size=state_size, action_size=action_size, **hp)
    if agent_type == "ppo":
        if hp.get("net") == "cnn":
            hp["grid_shape"] = grid_shape
        return PPOAgent(state_size=state_size, action_size=action_size, **hp)
    if agent_type == "drqn":
        return DRQNAgent(state_size=state_size, action_size=action_size,
                         grid_shape=grid_shape, **hp)
    if agent_type == "dtqn":
        return DTQNAgent(state_size=state_size, action_size=action_size,
                         grid_shape=grid_shape, **hp)
    raise ValueError(f"Unknown agent type: {agent_type}")


def train_agent(env: MazeEnvironment, agent, num_episodes: int, max_steps: int,
                bus: EventBus | None = None,
                policy_snapshot_every: int = 50,
                emit_steps: bool = True,
                emit_q_values: bool = False,
                random_start: bool = True):
    if bus is not None:
        bus.publish(RunEvent(kind="start",
                             info={"num_episodes": num_episodes,
                                   "max_steps": max_steps,
                                   "agent": type(agent).__name__}))

    for episode in range(num_episodes):
        state = env.reset(at_start=not random_start)
        if hasattr(agent, "on_episode_start"):
            agent.on_episode_start()
        total = 0.0
        length = 0
        success = False
        emit = bus is not None and emit_steps
        for step in range(max_steps):
            action = agent.move(state)
            qv = agent.q_values(state) if emit and emit_q_values else None
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total += float(reward)
            length += 1
            if emit:
                bus.publish(StepEvent(
                    episode=episode, step=step,
                    state=np.asarray(next_state),
                    position=env.agent_positions[0],
                    action=int(action), reward=float(reward), done=bool(done),
                    q_values=qv,
                ))
            state = next_state
            if done:
                success = reward > 0
                break

        if bus is not None:
            bus.publish(EpisodeEvent(
                episode=episode, total_reward=total, length=length,
                epsilon=float(getattr(agent, "epsilon", 0.0)),
                loss=getattr(agent, "last_loss", None),
                success=success,
            ))
            if policy_snapshot_every and (episode % policy_snapshot_every == 0):
                bus.publish(PolicyEvent(episode=episode,
                                        snapshot=agent.policy_snapshot()))

    if hasattr(agent, "flush"):
        agent.flush()
    if bus is not None:
        bus.publish(RunEvent(kind="end", info={}))
    return agent


def simulate_episode(env: MazeEnvironment, agent, max_steps: int,
                     at_start: bool = False
                     ) -> tuple[list[np.ndarray], list[tuple[int, int]], list[int], float]:
    state = env.reset(at_start=at_start) if at_start else env.reset()
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start()
    states = [state.copy()]
    positions: list[tuple[int, int]] = [env.agent_positions[0]]
    actions: list[int] = []
    total = 0.0
    for _ in range(max_steps):
        action = agent.move(state)
        next_state, reward, done, _ = env.step(action)
        states.append(next_state.copy())
        positions.append(env.agent_positions[0])
        actions.append(int(action))
        total += float(reward)
        state = next_state
        if done:
            break
    return states, positions, actions, total


def evaluate_agent(env: MazeEnvironment, agent, num_episodes: int, max_steps: int,
                   deterministic: bool = True
                   ) -> tuple[float, float, float]:
    rewards, lengths, successes = [], [], 0
    prev = getattr(agent, "deterministic", False)
    if deterministic:
        agent.set_deterministic(True)
    try:
        for _ in range(num_episodes):
            states, _, actions, total = simulate_episode(env, agent, max_steps)
            rewards.append(total)
            lengths.append(len(actions))
            if total > 0:
                successes += 1
    finally:
        if deterministic:
            agent.set_deterministic(prev)
    return float(np.mean(rewards)), float(np.mean(lengths)), successes / num_episodes
