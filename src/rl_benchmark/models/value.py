"""Value-side model components for critic-style baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import Tensor, nn

from rl_benchmark.models.policy import MLPBackbone, PolicyOutput, apply_orthogonal_initialization


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        self.value_head = nn.Linear(self.backbone.output_dim, 1)

        if orthogonal_init:
            apply_orthogonal_initialization(self.value_head)

    def forward(self, observations: Tensor) -> Tensor:
        features = self.backbone(observations)
        values = self.value_head(features)
        return values.squeeze(-1)


@dataclass(slots=True)
class ActorCriticOutput:
    policy: PolicyOutput
    values: Tensor


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "tanh",
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        self.shared_backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            orthogonal_init=orthogonal_init,
        )
        self.policy_head = nn.Linear(self.shared_backbone.output_dim, action_dim)
        self.value_head = nn.Linear(self.shared_backbone.output_dim, 1)

        if orthogonal_init:
            apply_orthogonal_initialization(self.policy_head)
            apply_orthogonal_initialization(self.value_head)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.shared_backbone(observations)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def act(self, observations: Tensor, deterministic: bool = False) -> ActorCriticOutput:
        logits, values = self.forward(observations)
        from torch.distributions import Categorical

        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        policy = PolicyOutput(
            logits=distribution.logits,
            actions=actions,
            log_probs=distribution.log_prob(actions),
            entropy=distribution.entropy(),
        )
        return ActorCriticOutput(policy=policy, values=values)
