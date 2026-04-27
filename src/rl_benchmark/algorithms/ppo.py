"""PPO implementation for discrete-action environments."""

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
class PPOBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    next_observation: np.ndarray


class PPOAlgorithm(RLAlgorithm):
    def __init__(
        self,
        config: dict[str, Any],
        observation_dim: int,
        action_dim: int,
        device: str = "auto",
    ) -> None:
        super().__init__(name="ppo", config=config, device=device)
        algorithm_config = config["algorithm"]
        network_config = config["network"]

        self.gamma = float(algorithm_config["gamma"])
        self.gae_lambda = float(algorithm_config["gae_lambda"])
        self.rollout_steps = int(algorithm_config["rollout_steps"])
        self.minibatch_size = int(algorithm_config["minibatch_size"])
        self.update_epochs = int(algorithm_config["update_epochs"])
        self.clip_coef = float(algorithm_config["clip_coef"])
        self.entropy_coef = float(algorithm_config.get("entropy_coef", 0.0))
        self.value_loss_coef = float(algorithm_config.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(algorithm_config.get("max_grad_norm", 0.5))
        self.target_kl = float(algorithm_config.get("target_kl", 0.02))

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

    def compute_advantages_and_returns(
        self,
        rewards: np.ndarray | Tensor,
        dones: np.ndarray | Tensor,
        values: np.ndarray | Tensor,
        next_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        value_tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        next_value = next_value.to(device=self.device, dtype=torch.float32).reshape(())

        advantages = torch.zeros_like(reward_tensor)
        last_advantage = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for index in range(reward_tensor.shape[0] - 1, -1, -1):
            next_non_terminal = 1.0 - done_tensor[index]
            next_state_value = next_value if index == reward_tensor.shape[0] - 1 else value_tensor[index + 1]
            delta = reward_tensor[index] + self.gamma * next_state_value * next_non_terminal - value_tensor[index]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages[index] = last_advantage

        returns = advantages + value_tensor
        return advantages, returns

    def update(self, batch: PPOBatch | dict[str, np.ndarray]) -> dict[str, float]:
        rollout_batch = self._coerce_batch(batch)
        observations = self.prepare_tensor(rollout_batch.observations)
        actions = torch.as_tensor(rollout_batch.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(rollout_batch.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(rollout_batch.values, dtype=torch.float32, device=self.device)
        next_observation = self.prepare_tensor(rollout_batch.next_observation)
        dones = torch.as_tensor(rollout_batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_values = self.actor_critic(next_observation)
            next_value = next_values.squeeze(0) * (1.0 - dones[-1])

        advantages, returns = self.compute_advantages_and_returns(
            rewards=rollout_batch.rewards,
            dones=rollout_batch.dones,
            values=old_values,
            next_value=next_value,
        )
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        grad_norms: list[float] = []
        approx_kls: list[float] = []

        batch_size = observations.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        indices = torch.arange(batch_size, device=self.device)

        for _ in range(self.update_epochs):
            shuffled = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                batch_indices = shuffled[start : start + minibatch_size]

                logits, new_values = self.actor_critic(observations[batch_indices])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[batch_indices])
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                unclipped = ratio * advantages[batch_indices]
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages[batch_indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * torch.square(returns[batch_indices] - new_values).mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_probs[batch_indices] - new_log_probs).mean()
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                grad_norms.append(float(grad_norm.detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))

                if approx_kl.detach().abs().item() > self.target_kl:
                    break

            if approx_kls and abs(approx_kls[-1]) > self.target_kl:
                break

        return {
            "policy_loss": float(np.mean(policy_losses, dtype=np.float64)),
            "value_loss": float(np.mean(value_losses, dtype=np.float64)),
            "entropy": float(np.mean(entropies, dtype=np.float64)),
            "grad_norm": float(np.mean(grad_norms, dtype=np.float64)),
            "approx_kl": float(np.mean(approx_kls, dtype=np.float64)),
            "rollout_return": float(np.sum(rollout_batch.rewards, dtype=np.float64)),
            "rollout_length": float(len(rollout_batch.rewards)),
        }

    def _coerce_batch(self, batch: PPOBatch | dict[str, np.ndarray]) -> PPOBatch:
        if isinstance(batch, PPOBatch):
            return batch

        return PPOBatch(
            observations=np.asarray(batch["observations"], dtype=np.float32),
            actions=np.asarray(batch["actions"], dtype=np.int64),
            rewards=np.asarray(batch["rewards"], dtype=np.float32),
            dones=np.asarray(batch["dones"], dtype=np.float32),
            log_probs=np.asarray(batch["log_probs"], dtype=np.float32),
            values=np.asarray(batch["values"], dtype=np.float32),
            next_observation=np.asarray(batch["next_observation"], dtype=np.float32),
        )
