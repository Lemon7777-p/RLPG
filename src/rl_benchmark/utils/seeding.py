"""Deterministic seeding helpers used across training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, bool]:
    """Seed Python, NumPy, and Torch when it is available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    try:
        import torch
    except ImportError:
        return {"python": True, "numpy": True, "torch": torch_seeded}

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch_seeded = True
    return {"python": True, "numpy": True, "torch": torch_seeded}


def seed_space(space: Any, seed: int | None) -> None:
    if seed is None:
        return

    if hasattr(space, "seed"):
        space.seed(seed)


def seed_env_spaces(env: Any, seed: int | None) -> None:
    if seed is None:
        return

    if hasattr(env, "action_space"):
        seed_space(env.action_space, seed)

    if hasattr(env, "observation_space"):
        seed_space(env.observation_space, seed)
