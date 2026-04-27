"""REINFORCE implementation for discrete-action environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from rl_benchmark.algorithms.base import RLAlgorithm
from rl_benchmark.models import DiscretePolicyNetwork


@dataclass(slots=True)
class EpisodeBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


class ReinforceAlgorithm(RLAlgorithm):
    def __init__(
        self,
        config: dict[str, Any],
        observation_dim: int,
        action_dim: int,
        device: str = "auto",
    ) -> None:
        super().__init__(name="reinforce", config=config, device=device)
        algorithm_config = config["algorithm"]
        network_config = config["network"]

        self.gamma = float(algorithm_config["gamma"])
        self.entropy_coef = float(algorithm_config.get("entropy_coef", 0.0))
        self.normalize_returns = bool(algorithm_config.get("normalize_returns", True))
        self.max_grad_norm = float(algorithm_config.get("max_grad_norm", 1.0))

        self.policy = DiscretePolicyNetwork(
            input_dim=observation_dim,
            action_dim=action_dim,
            hidden_sizes=network_config["hidden_sizes"],
            activation=network_config.get("activation", "tanh"),
            orthogonal_init=bool(network_config.get("orthogonal_init", True)),
        )
        self.finalize_setup()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(algorithm_config["learning_rate"]),
        )

    def act(self, observations: np.ndarray | Tensor, deterministic: bool = False):
        observation_tensor = self.prepare_tensor(observations)
        return self.policy.act(observation_tensor, deterministic=deterministic)

    def compute_discounted_returns(self, rewards: np.ndarray | Tensor) -> Tensor:
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        returns = torch.zeros_like(reward_tensor)
        running_return = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for index in range(reward_tensor.shape[0] - 1, -1, -1):
            running_return = reward_tensor[index] + self.gamma * running_return
            returns[index] = running_return

        if self.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / returns.std().clamp_min(1e-8)

        return returns

    def update(self, batch: EpisodeBatch | dict[str, np.ndarray]) -> dict[str, float]:
        episode_batch = self._coerce_batch(batch)
        observations = self.prepare_tensor(episode_batch.observations)
        actions = torch.as_tensor(episode_batch.actions, dtype=torch.long, device=self.device)
        returns = self.compute_discounted_returns(episode_batch.rewards)

        distribution = self.policy.distribution(observations)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        policy_loss = -(log_probs * returns).mean()
        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item()),
            "episode_return": float(np.sum(episode_batch.rewards, dtype=np.float64)),
            "episode_length": float(episode_batch.rewards.shape[0]),
        }

    def _coerce_batch(self, batch: EpisodeBatch | dict[str, np.ndarray]) -> EpisodeBatch:
        if isinstance(batch, EpisodeBatch):
            return batch

        return EpisodeBatch(
            observations=np.asarray(batch["observations"], dtype=np.float32),
            actions=np.asarray(batch["actions"], dtype=np.int64),
            rewards=np.asarray(batch["rewards"], dtype=np.float32),
        )
