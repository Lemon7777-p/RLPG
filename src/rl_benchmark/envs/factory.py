"""Environment construction helpers for reproducible experiment setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from rl_benchmark.envs.wrappers import apply_common_wrappers
from rl_benchmark.utils.seeding import seed_env_spaces


Env = gym.Env

ENV_ID_ALIASES = {
    "LunarLander-v2": "LunarLander-v3",
}


@dataclass(frozen=True, slots=True)
class EnvConfig:
    env_id: str
    seed: int | None = None
    render_mode: str | None = None
    record_episode_statistics: bool = True
    disable_env_checker: bool = True


@dataclass(slots=True)
class CreatedEnv:
    env: Env
    observation: Any
    info: dict[str, Any]
    requested_env_id: str
    resolved_env_id: str


def resolve_env_id(env_id: str) -> str:
    """Map requested benchmark IDs to the Gymnasium runtime IDs available here."""
    return ENV_ID_ALIASES.get(env_id, env_id)


def build_env(config: EnvConfig | str, **overrides: Any) -> Env:
    """Create an environment and apply the shared wrapper stack."""
    env_config = _resolve_config(config, **overrides)
    env = gym.make(
        resolve_env_id(env_config.env_id),
        render_mode=env_config.render_mode,
        disable_env_checker=env_config.disable_env_checker,
    )

    if env_config.record_episode_statistics:
        env = apply_common_wrappers(env)

    seed_env_spaces(env, env_config.seed)
    return env


def make_env(config: EnvConfig | str, **overrides: Any) -> CreatedEnv:
    """Create and reset an environment in one step for deterministic startup."""
    env_config = _resolve_config(config, **overrides)
    resolved_env_id = resolve_env_id(env_config.env_id)
    env = build_env(env_config)
    observation, info = env.reset(seed=env_config.seed)
    return CreatedEnv(
        env=env,
        observation=observation,
        info=info,
        requested_env_id=env_config.env_id,
        resolved_env_id=resolved_env_id,
    )


def make_train_and_eval_envs(
    env_id: str,
    seed: int,
    eval_seed_offset: int = 10_000,
    **overrides: Any,
) -> tuple[CreatedEnv, CreatedEnv]:
    """Create paired train and evaluation environments with disjoint seeds."""
    train_env = make_env(env_id, seed=seed, **overrides)
    eval_env = make_env(env_id, seed=seed + eval_seed_offset, **overrides)
    return train_env, eval_env


def _resolve_config(config: EnvConfig | str, **overrides: Any) -> EnvConfig:
    if isinstance(config, EnvConfig):
        config_values = {
            "env_id": config.env_id,
            "seed": config.seed,
            "render_mode": config.render_mode,
            "record_episode_statistics": config.record_episode_statistics,
            "disable_env_checker": config.disable_env_checker,
        }
        config_values.update(overrides)
        return EnvConfig(**config_values)

    return EnvConfig(env_id=config, **overrides)
