"""Common abstractions shared by all benchmark algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device

    normalized = device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(normalized)


class RLAlgorithm(nn.Module, ABC):
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.device = resolve_device(device)

    def finalize_setup(self) -> None:
        self.to(self.device)

    def prepare_tensor(
        self,
        observations: np.ndarray | Tensor | list[float] | list[list[float]],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        tensor = torch.as_tensor(observations, dtype=dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def set_training(self, training: bool = True) -> None:
        self.train(training)

    def checkpoint_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        return payload

    def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        optimizer = getattr(self, "optimizer", None)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "config": self.config,
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        payload.update(self.checkpoint_state())
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.restore_checkpoint_state(checkpoint)
        return checkpoint

    @abstractmethod
    def act(self, observations: np.ndarray | Tensor, deterministic: bool = False) -> Any:
        """Select actions for a batch of observations."""

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Perform one optimization step and return logging metrics."""
