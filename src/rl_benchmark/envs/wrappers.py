"""Common Gymnasium wrappers used by all experiments."""

from __future__ import annotations

import gymnasium as gym


Env = gym.Env


def apply_common_wrappers(env: Env) -> Env:
    """Attach the standard wrapper stack for benchmark runs."""
    return gym.wrappers.RecordEpisodeStatistics(env)
