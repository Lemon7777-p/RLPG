"""Shared policy-side model components for discrete-action environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def resolve_activation(name: str) -> type[nn.Module]:
    normalized_name = name.strip().lower()
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }

    if normalized_name not in activation_map:
        raise ValueError(f"Unsupported activation: {name}")

    return activation_map[normalized_name]


def apply_orthogonal_initialization(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "tanh",
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size")

        activation_cls = resolve_activation(activation)
        layers: list[nn.Module] = []
        feature_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(activation_cls())
            feature_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = feature_dim

        if orthogonal_init:
            self.apply(apply_orthogonal_initialization)

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations)


@dataclass(slots=True)
class PolicyOutput:
    logits: Tensor
    actions: Tensor
    log_probs: Tensor
    entropy: Tensor


class DiscretePolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "tanh",
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            orthogonal_init=orthogonal_init,
        )
        self.policy_head = nn.Linear(self.backbone.output_dim, action_dim)

        if orthogonal_init:
            apply_orthogonal_initialization(self.policy_head)

    def forward(self, observations: Tensor) -> Tensor:
        features = self.backbone(observations)
        return self.policy_head(features)

    def distribution(self, observations: Tensor) -> Categorical:
        logits = self.forward(observations)
        return Categorical(logits=logits)

    def act(self, observations: Tensor, deterministic: bool = False) -> PolicyOutput:
        distribution = self.distribution(observations)
        logits = distribution.logits
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return PolicyOutput(
            logits=logits,
            actions=actions,
            log_probs=log_probs,
            entropy=entropy,
        )
