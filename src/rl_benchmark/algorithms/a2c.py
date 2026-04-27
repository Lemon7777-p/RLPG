"""A2C implementation for discrete-action environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from rl_benchmark.algorithms.base import RLAlgorithm
from rl_benchmark.models import ActorCriticNetwork


@dataclass(slots=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_observation: np.ndarray


class A2CAlgorithm(RLAlgorithm):
    def __init__(
        self,
        config: dict[str, Any],
        observation_dim: int,
        action_dim: int,
        device: str = "auto",
    ) -> None:
        super().__init__(name="a2c", config=config, device=device)
        algorithm_config = config["algorithm"]
        network_config = config["network"]

        self.gamma = float(algorithm_config["gamma"])
        self.rollout_steps = int(algorithm_config["rollout_steps"])
        self.entropy_coef = float(algorithm_config.get("entropy_coef", 0.0))
        self.value_loss_coef = float(algorithm_config.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(algorithm_config.get("max_grad_norm", 0.5))

        self.actor_critic = ActorCriticNetwork(
            input_dim=observation_dim,
            action_dim=action_dim,
            hidden_sizes=network_config["hidden_sizes"],
            activation=network_config.get("activation", "tanh"),
            orthogonal_init=bool(network_config.get("orthogonal_init", True)),
        )
        self.finalize_setup()
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=float(algorithm_config["learning_rate"]),
        )

    def act(self, observations: np.ndarray | Tensor, deterministic: bool = False):
        observation_tensor = self.prepare_tensor(observations)
        return self.actor_critic.act(observation_tensor, deterministic=deterministic)

    def compute_returns(
        self,
        rewards: np.ndarray | Tensor,
        dones: np.ndarray | Tensor,
        bootstrap_value: Tensor,
    ) -> Tensor:
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        returns = torch.zeros_like(reward_tensor)
        running_return = bootstrap_value.to(device=self.device, dtype=torch.float32).reshape(())

        for index in range(reward_tensor.shape[0] - 1, -1, -1):
            running_return = reward_tensor[index] + self.gamma * running_return * (1.0 - done_tensor[index])
            returns[index] = running_return

        return returns

    def update(self, batch: RolloutBatch | dict[str, np.ndarray]) -> dict[str, float]:
        rollout_batch = self._coerce_batch(batch)
        observations = self.prepare_tensor(rollout_batch.observations)
        actions = torch.as_tensor(rollout_batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rollout_batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(rollout_batch.dones, dtype=torch.float32, device=self.device)
        next_observation = self.prepare_tensor(rollout_batch.next_observation)

        logits, values = self.actor_critic(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        with torch.no_grad():
            _, next_values = self.actor_critic(next_observation)
            bootstrap_value = next_values.squeeze(0) * (1.0 - dones[-1])

        returns = self.compute_returns(rewards, dones, bootstrap_value)
        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * torch.square(returns - values).mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item()),
            "rollout_return": float(rewards.sum().detach().cpu().item()),
            "rollout_length": float(rewards.shape[0]),
        }

    def _coerce_batch(self, batch: RolloutBatch | dict[str, np.ndarray]) -> RolloutBatch:
        if isinstance(batch, RolloutBatch):
            return batch

        return RolloutBatch(
            observations=np.asarray(batch["observations"], dtype=np.float32),
            actions=np.asarray(batch["actions"], dtype=np.int64),
            rewards=np.asarray(batch["rewards"], dtype=np.float32),
            dones=np.asarray(batch["dones"], dtype=np.float32),
            next_observation=np.asarray(batch["next_observation"], dtype=np.float32),
        )
