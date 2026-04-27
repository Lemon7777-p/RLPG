"""Training bootstrap utilities for single-run experiment setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np

from rl_benchmark.algorithms import (
    A2CAlgorithm,
    PPOAlgorithm,
    EpisodeBatch,
    PPOBatch,
    ReinforceAlgorithm,
    RolloutBatch,
)
from rl_benchmark.config import load_run_config
from rl_benchmark.envs.factory import CreatedEnv, make_train_and_eval_envs
from rl_benchmark.logging.schema import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MetricRecord,
    RunManifest,
    append_metrics,
    checkpoint_dir_for,
    read_manifest,
    read_metrics,
    resolve_results_root,
    write_manifest,
    write_metrics,
)
from rl_benchmark.utils.seeding import set_global_seed


@dataclass(slots=True)
class RunContext:
    algorithm_name: str
    env_id: str
    seed: int
    config: dict[str, Any]
    train_env: CreatedEnv
    eval_env: CreatedEnv
    observation_dim: int
    action_dim: int
    output_dir: Path

    def close(self) -> None:
        self.train_env.env.close()
        self.eval_env.env.close()


@dataclass(slots=True)
class TrainState:
    observation: np.ndarray
    episode_return: float = 0.0
    episode_length: int = 0


@dataclass(slots=True)
class TrainingResult:
    run_id: str
    output_dir: Path
    manifest: RunManifest
    metrics: list[MetricRecord]
    latest_checkpoint: Path | None = None


def infer_observation_dim(space: gym.Space[Any]) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape, dtype=np.int64))

    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)

    raise TypeError(f"Unsupported observation space: {space}")


def infer_action_dim(space: gym.Space[Any]) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)

    raise TypeError(f"Unsupported action space: {space}")


def build_run_id(algorithm_name: str, env_id: str, seed: int) -> str:
    safe_env_id = env_id.replace("/", "-")
    return f"{algorithm_name}_{safe_env_id}_seed{seed}"


def prepare_run_context(
    algorithm_name: str,
    env_id: str,
    seed: int,
    results_root: str | Path | None = None,
) -> RunContext:
    run_config = load_run_config(algorithm_name, env_id)
    deterministic = bool(run_config["runtime"].get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    train_env, eval_env = make_train_and_eval_envs(env_id, seed=seed)
    output_root = resolve_results_root(results_root or run_config["project"]["output_root"])
    output_dir = output_root / build_run_id(algorithm_name, env_id, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        algorithm_name=algorithm_name,
        env_id=env_id,
        seed=seed,
        config=run_config,
        train_env=train_env,
        eval_env=eval_env,
        observation_dim=infer_observation_dim(train_env.env.observation_space),
        action_dim=infer_action_dim(train_env.env.action_space),
        output_dir=output_dir,
    )


def create_algorithm(run_context: RunContext, device: str = "auto"):
    algorithm_map = {
        "reinforce": ReinforceAlgorithm,
        "a2c": A2CAlgorithm,
        "ppo": PPOAlgorithm,
    }
    try:
        algorithm_cls = algorithm_map[run_context.algorithm_name]
    except KeyError as error:
        raise KeyError(f"Unsupported algorithm: {run_context.algorithm_name}") from error

    return algorithm_cls(
        config=run_context.config,
        observation_dim=run_context.observation_dim,
        action_dim=run_context.action_dim,
        device=device,
    )


def run_training_job(
    algorithm_name: str,
    env_id: str,
    seed: int,
    *,
    device: str = "auto",
    train_steps: int | None = None,
    eval_episodes: int | None = None,
    results_root: str | Path | None = None,
    notes: str = "",
    persist_every_update: bool = True,
    checkpoint_interval_steps: int | None = None,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
    log_progress: bool = False,
    progress_interval_steps: int | None = None,
) -> TrainingResult:
    context = prepare_run_context(algorithm_name, env_id, seed, results_root=results_root)
    algorithm = create_algorithm(context, device=device)
    max_steps = int(train_steps or context.config["environment"]["train_steps"])
    eval_interval_steps = int(
        context.config["environment"].get(
            "eval_interval_steps",
            context.config["runtime"].get("eval_interval_steps", max_steps),
        )
    )
    eval_episodes = int(eval_episodes or context.config["evaluation"].get("episodes", 10))
    deterministic_eval = bool(context.config["evaluation"].get("deterministic", True))
    checkpoint_interval_steps = int(
        checkpoint_interval_steps
        if checkpoint_interval_steps is not None
        else context.config["runtime"].get("checkpoint_interval_steps", 0)
    )
    checkpoint_interval_steps = max(0, checkpoint_interval_steps)

    run_id = build_run_id(algorithm_name, env_id, seed)
    if resume or checkpoint_path is not None:
        (
            manifest,
            metrics,
            total_steps,
            total_updates,
            next_eval_step,
            train_state,
            elapsed_time_offset,
        ) = _load_resume_state(
            context,
            algorithm,
            eval_interval_steps=eval_interval_steps,
            checkpoint_path=checkpoint_path,
            notes=notes,
        )
    else:
        manifest = RunManifest(
            run_id=run_id,
            algorithm_name=algorithm_name,
            env_id=env_id,
            runtime_env_id=context.config["run"]["runtime_env_id"],
            seed=seed,
            status="running",
            total_steps=0,
            total_updates=0,
            notes=notes,
            config_snapshot=context.config,
        )
        write_manifest(manifest, context.output_dir.parent)
        metrics = []
        total_steps = 0
        total_updates = 0
        next_eval_step = _next_threshold(0, eval_interval_steps)
        train_state = TrainState(observation=np.asarray(context.train_env.observation, dtype=np.float32))
        elapsed_time_offset = 0.0

    start_time = perf_counter() - elapsed_time_offset
    next_checkpoint_step = _next_threshold(total_steps, checkpoint_interval_steps)
    resolved_progress_interval = _resolve_progress_interval(
        progress_interval_steps,
        eval_interval_steps=eval_interval_steps,
        checkpoint_interval_steps=checkpoint_interval_steps,
        max_steps=max_steps,
    )
    next_progress_step = _next_threshold(total_steps, resolved_progress_interval) if log_progress else None
    latest_checkpoint: Path | None = None

    try:
        while total_steps < max_steps:
            remaining_steps = max_steps - total_steps
            batch, train_state, completed_returns, completed_lengths = _collect_batch(
                algorithm,
                context.train_env.env,
                train_state,
                remaining_steps=remaining_steps,
            )
            update_metrics = algorithm.update(batch)
            batch_steps = _batch_size(batch)
            total_steps += batch_steps
            total_updates += 1

            eval_return: float | None = None
            if total_steps >= next_eval_step or total_steps >= max_steps:
                eval_return = evaluate_policy(
                    algorithm,
                    context.eval_env.env,
                    episodes=eval_episodes,
                    deterministic=deterministic_eval,
                )
                next_eval_step += eval_interval_steps

            train_return, episode_length = _resolve_training_observation(
                update_metrics,
                completed_returns,
                completed_lengths,
            )
            metric_record = MetricRecord(
                step=int(total_steps),
                update=total_updates,
                wall_time_sec=float(perf_counter() - start_time),
                train_episode_return=train_return,
                eval_episode_return=eval_return,
                episode_length=episode_length,
                policy_loss=_optional_metric(update_metrics, "policy_loss", "loss"),
                value_loss=_optional_metric(update_metrics, "value_loss"),
                entropy=_optional_metric(update_metrics, "entropy"),
                grad_norm=_optional_metric(update_metrics, "grad_norm"),
                approx_kl=_optional_metric(update_metrics, "approx_kl"),
            )
            metrics.append(metric_record)

            manifest.total_steps = total_steps
            manifest.total_updates = total_updates
            if checkpoint_interval_steps > 0 and next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
                latest_checkpoint = _save_training_checkpoint(
                    algorithm,
                    context=context,
                    total_steps=total_steps,
                    total_updates=total_updates,
                    next_eval_step=next_eval_step,
                    train_state=train_state,
                    wall_time_sec=metric_record.wall_time_sec,
                    label="update",
                )
                manifest.latest_checkpoint = _relative_to_run_dir(latest_checkpoint, context.output_dir)
                manifest.checkpoint_count += 1
                next_checkpoint_step = _next_threshold(total_steps, checkpoint_interval_steps)
            write_manifest(manifest, context.output_dir.parent)
            if persist_every_update:
                append_metrics(run_id, [metric_record], context.output_dir.parent)
            if log_progress and next_progress_step is not None and total_steps >= next_progress_step:
                _print_progress(
                    run_id,
                    total_steps=total_steps,
                    max_steps=max_steps,
                    total_updates=total_updates,
                    metric_record=metric_record,
                )
                next_progress_step = _next_threshold(total_steps, resolved_progress_interval)

        manifest.status = "completed"
        latest_checkpoint = _save_training_checkpoint(
            algorithm,
            context=context,
            total_steps=total_steps,
            total_updates=total_updates,
            next_eval_step=next_eval_step,
            train_state=train_state,
            wall_time_sec=float(perf_counter() - start_time),
            label="final",
        )
        manifest.latest_checkpoint = _relative_to_run_dir(latest_checkpoint, context.output_dir)
        manifest.checkpoint_count += 1
        write_manifest(manifest, context.output_dir.parent)
        if not persist_every_update:
            write_metrics(run_id, metrics, context.output_dir.parent)
        return TrainingResult(
            run_id=run_id,
            output_dir=context.output_dir,
            manifest=manifest,
            metrics=metrics,
            latest_checkpoint=latest_checkpoint,
        )
    except KeyboardInterrupt as error:
        manifest.status = "failed"
        manifest.failure_message = f"{type(error).__name__}: {error}"
        write_manifest(manifest, context.output_dir.parent)
        raise
    except Exception as error:
        manifest.status = "failed"
        manifest.failure_message = f"{type(error).__name__}: {error}"
        write_manifest(manifest, context.output_dir.parent)
        if metrics and not persist_every_update:
            write_metrics(run_id, metrics, context.output_dir.parent)
        raise
    finally:
        context.close()


def resolve_resume_checkpoint(run_dir: Path, checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = read_manifest(manifest_path)
        if manifest.latest_checkpoint:
            return run_dir / manifest.latest_checkpoint

    checkpoint_dir = checkpoint_dir_for(run_dir.name, run_dir.parent)
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files were found for run: {run_dir.name}")
    return checkpoint_files[-1]


def evaluate_policy(
    algorithm: ReinforceAlgorithm | A2CAlgorithm | PPOAlgorithm,
    env: gym.Env[Any],
    *,
    episodes: int,
    deterministic: bool,
) -> float:
    episode_returns: list[float] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = _action_from_output(algorithm.act(observation, deterministic=deterministic))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        episode_returns.append(total_reward)

    return float(np.mean(episode_returns, dtype=np.float64))


def _collect_batch(
    algorithm: ReinforceAlgorithm | A2CAlgorithm | PPOAlgorithm,
    env: gym.Env[Any],
    train_state: TrainState,
    *,
    remaining_steps: int,
) -> tuple[EpisodeBatch | RolloutBatch | PPOBatch, TrainState, list[float], list[int]]:
    if isinstance(algorithm, ReinforceAlgorithm):
        return _collect_reinforce_episode(algorithm, env, train_state)

    if isinstance(algorithm, A2CAlgorithm):
        return _collect_actor_critic_rollout(algorithm, env, train_state, rollout_steps=min(algorithm.rollout_steps, remaining_steps))

    if isinstance(algorithm, PPOAlgorithm):
        return _collect_ppo_rollout(algorithm, env, train_state, rollout_steps=min(algorithm.rollout_steps, remaining_steps))

    raise TypeError(f"Unsupported algorithm type: {type(algorithm)!r}")


def _collect_reinforce_episode(
    algorithm: ReinforceAlgorithm,
    env: gym.Env[Any],
    train_state: TrainState,
) -> tuple[EpisodeBatch, TrainState, list[float], list[int]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    observation = np.asarray(train_state.observation, dtype=np.float32)
    episode_return = train_state.episode_return
    episode_length = train_state.episode_length

    while True:
        action = _action_from_output(algorithm.act(observation, deterministic=False))
        next_observation, reward, terminated, truncated, _ = env.step(action)

        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            reset_observation, _ = env.reset()
            batch = EpisodeBatch(
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
            )
            next_state = TrainState(observation=np.asarray(reset_observation, dtype=np.float32))
            return batch, next_state, [episode_return], [episode_length]

        observation = np.asarray(next_observation, dtype=np.float32)


def _collect_actor_critic_rollout(
    algorithm: A2CAlgorithm,
    env: gym.Env[Any],
    train_state: TrainState,
    *,
    rollout_steps: int,
) -> tuple[RolloutBatch, TrainState, list[float], list[int]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    observation = np.asarray(train_state.observation, dtype=np.float32)
    episode_return = train_state.episode_return
    episode_length = train_state.episode_length

    for _ in range(rollout_steps):
        output = algorithm.act(observation, deterministic=False)
        action = _action_from_output(output)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))

        episode_return += float(reward)
        episode_length += 1

        if done:
            completed_returns.append(episode_return)
            completed_lengths.append(episode_length)
            reset_observation, _ = env.reset()
            observation = np.asarray(reset_observation, dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
        else:
            observation = np.asarray(next_observation, dtype=np.float32)

    batch = RolloutBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        next_observation=np.asarray(observation, dtype=np.float32),
    )
    next_state = TrainState(
        observation=np.asarray(observation, dtype=np.float32),
        episode_return=episode_return,
        episode_length=episode_length,
    )
    return batch, next_state, completed_returns, completed_lengths


def _collect_ppo_rollout(
    algorithm: PPOAlgorithm,
    env: gym.Env[Any],
    train_state: TrainState,
    *,
    rollout_steps: int,
) -> tuple[PPOBatch, TrainState, list[float], list[int]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    log_probs: list[float] = []
    values: list[float] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    observation = np.asarray(train_state.observation, dtype=np.float32)
    episode_return = train_state.episode_return
    episode_length = train_state.episode_length

    for _ in range(rollout_steps):
        output = algorithm.act(observation, deterministic=False)
        action = _action_from_output(output)
        log_prob = float(output.policy.log_probs.squeeze(0).detach().cpu().item())
        value = float(output.values.squeeze(0).detach().cpu().item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(observation)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        episode_return += float(reward)
        episode_length += 1

        if done:
            completed_returns.append(episode_return)
            completed_lengths.append(episode_length)
            reset_observation, _ = env.reset()
            observation = np.asarray(reset_observation, dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
        else:
            observation = np.asarray(next_observation, dtype=np.float32)

    batch = PPOBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_observation=np.asarray(observation, dtype=np.float32),
    )
    next_state = TrainState(
        observation=np.asarray(observation, dtype=np.float32),
        episode_return=episode_return,
        episode_length=episode_length,
    )
    return batch, next_state, completed_returns, completed_lengths


def _action_from_output(output: Any) -> int:
    if hasattr(output, "policy"):
        action_tensor = output.policy.actions
    else:
        action_tensor = output.actions
    return int(action_tensor.squeeze(0).detach().cpu().item())


def _resolve_training_observation(
    update_metrics: dict[str, float],
    completed_returns: list[float],
    completed_lengths: list[int],
) -> tuple[float | None, float | None]:
    if completed_returns:
        return (
            float(np.mean(completed_returns, dtype=np.float64)),
            float(np.mean(completed_lengths, dtype=np.float64)),
        )

    train_return = update_metrics.get("episode_return", update_metrics.get("rollout_return"))
    episode_length = update_metrics.get("episode_length", update_metrics.get("rollout_length"))
    return (
        float(train_return) if train_return is not None else None,
        float(episode_length) if episode_length is not None else None,
    )


def _optional_metric(update_metrics: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in update_metrics:
            return float(update_metrics[key])
    return None


def _batch_size(batch: EpisodeBatch | RolloutBatch | PPOBatch) -> int:
    return int(len(batch.rewards))


def _load_resume_state(
    context: RunContext,
    algorithm: ReinforceAlgorithm | A2CAlgorithm | PPOAlgorithm,
    *,
    eval_interval_steps: int,
    checkpoint_path: str | Path | None,
    notes: str,
) -> tuple[RunManifest, list[MetricRecord], int, int, int, TrainState, float]:
    manifest_path = context.output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot resume run without a manifest: {manifest_path}")

    manifest = read_manifest(manifest_path)
    resolved_checkpoint = resolve_resume_checkpoint(context.output_dir, checkpoint_path)
    checkpoint = algorithm.load_checkpoint(resolved_checkpoint)
    metrics_path = context.output_dir / manifest.metrics_file
    metrics = read_metrics(metrics_path)
    metadata = checkpoint.get("metadata", {})
    train_state_payload = metadata.get("train_state", {})
    observation = train_state_payload.get("observation", context.train_env.observation)
    train_state = TrainState(
        observation=np.asarray(observation, dtype=np.float32),
        episode_return=float(train_state_payload.get("episode_return", 0.0)),
        episode_length=int(train_state_payload.get("episode_length", 0)),
    )
    total_steps = int(metadata.get("total_steps", manifest.total_steps))
    total_updates = int(metadata.get("total_updates", manifest.total_updates))
    next_eval_step = int(metadata.get("next_eval_step", _next_threshold(total_steps, eval_interval_steps) or eval_interval_steps))
    elapsed_time_offset = float(metrics[-1].wall_time_sec if metrics else metadata.get("wall_time_sec", 0.0))

    manifest.status = "running"
    manifest.resumed_from = _relative_to_run_dir(resolved_checkpoint, context.output_dir)
    if notes:
        manifest.notes = notes
    return manifest, metrics, total_steps, total_updates, next_eval_step, train_state, elapsed_time_offset


def _save_training_checkpoint(
    algorithm: ReinforceAlgorithm | A2CAlgorithm | PPOAlgorithm,
    *,
    context: RunContext,
    total_steps: int,
    total_updates: int,
    next_eval_step: int,
    train_state: TrainState,
    wall_time_sec: float,
    label: str,
) -> Path:
    checkpoint_dir = checkpoint_dir_for(context.output_dir.name, context.output_dir.parent)
    checkpoint_name = f"{label}_update_{total_updates:06d}_step_{total_steps:08d}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    metadata = {
        "run_id": context.output_dir.name,
        "total_steps": total_steps,
        "total_updates": total_updates,
        "next_eval_step": next_eval_step,
        "wall_time_sec": wall_time_sec,
        "train_state": {
            "observation": train_state.observation.tolist(),
            "episode_return": train_state.episode_return,
            "episode_length": train_state.episode_length,
        },
    }
    return algorithm.save_checkpoint(checkpoint_path, metadata=metadata)


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _next_threshold(current_value: int, interval: int) -> int | None:
    if interval <= 0:
        return None
    return ((current_value // interval) + 1) * interval


def _resolve_progress_interval(
    requested_interval: int | None,
    *,
    eval_interval_steps: int,
    checkpoint_interval_steps: int,
    max_steps: int,
) -> int:
    if requested_interval is not None:
        return max(1, int(requested_interval))
    if checkpoint_interval_steps > 0:
        return checkpoint_interval_steps
    if eval_interval_steps > 0:
        return eval_interval_steps
    return max_steps


def _print_progress(
    run_id: str,
    *,
    total_steps: int,
    max_steps: int,
    total_updates: int,
    metric_record: MetricRecord,
) -> None:
    eval_text = "n/a" if metric_record.eval_episode_return is None else f"{metric_record.eval_episode_return:.2f}"
    train_text = "n/a" if metric_record.train_episode_return is None else f"{metric_record.train_episode_return:.2f}"
    print(
        f"Progress {run_id}: steps={total_steps}/{max_steps}, updates={total_updates}, train_return={train_text}, eval_return={eval_text}, wall_time_sec={metric_record.wall_time_sec:.2f}"
    )
