"""Helpers for building smoke-verification job plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_ALGORITHMS = ("reinforce", "a2c", "ppo")
DEFAULT_MAIN_ENVIRONMENTS = ("Acrobot-v1", "LunarLander-v2")
DEFAULT_SEEDS = (7, 11)


@dataclass(frozen=True, slots=True)
class VerificationJob:
    algorithm_name: str
    env_id: str
    seed: int
    train_steps: int
    checkpoint_interval_steps: int


def build_smoke_verification_jobs(
    *,
    algorithms: Iterable[str] = DEFAULT_ALGORITHMS,
    envs: Iterable[str] = DEFAULT_MAIN_ENVIRONMENTS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    reinforce_train_steps: int = 1,
    actor_critic_train_steps: int = 16,
) -> list[VerificationJob]:
    jobs: list[VerificationJob] = []
    for env_id in envs:
        for algorithm_name in algorithms:
            train_steps = reinforce_train_steps if algorithm_name == "reinforce" else actor_critic_train_steps
            checkpoint_interval_steps = max(8, train_steps // 2)
            for seed in seeds:
                jobs.append(
                    VerificationJob(
                        algorithm_name=algorithm_name,
                        env_id=env_id,
                        seed=int(seed),
                        train_steps=int(train_steps),
                        checkpoint_interval_steps=int(checkpoint_interval_steps),
                    )
                )
    return jobs