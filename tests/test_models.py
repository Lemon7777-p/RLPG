from __future__ import annotations

import unittest

import torch

from rl_benchmark.models import ActorCriticNetwork, DiscretePolicyNetwork, ValueNetwork


class PolicyNetworkTests(unittest.TestCase):
    def test_discrete_policy_outputs_expected_shapes(self) -> None:
        model = DiscretePolicyNetwork(
            input_dim=4,
            action_dim=2,
            hidden_sizes=[64, 64],
            activation="tanh",
        )
        observations = torch.randn(3, 4)

        output = model.act(observations)

        self.assertEqual(tuple(output.logits.shape), (3, 2))
        self.assertEqual(tuple(output.actions.shape), (3,))
        self.assertEqual(tuple(output.log_probs.shape), (3,))
        self.assertEqual(tuple(output.entropy.shape), (3,))

    def test_discrete_policy_deterministic_action_uses_argmax(self) -> None:
        model = DiscretePolicyNetwork(
            input_dim=4,
            action_dim=2,
            hidden_sizes=[8],
            activation="relu",
        )
        observations = torch.ones(2, 4)

        logits = model(observations)
        output = model.act(observations, deterministic=True)

        self.assertTrue(torch.equal(output.actions, logits.argmax(dim=-1)))


class ValueNetworkTests(unittest.TestCase):
    def test_value_network_outputs_batch_vector(self) -> None:
        model = ValueNetwork(
            input_dim=6,
            hidden_sizes=[32, 16],
            activation="tanh",
        )
        observations = torch.randn(5, 6)

        values = model(observations)

        self.assertEqual(tuple(values.shape), (5,))

    def test_actor_critic_outputs_policy_and_values(self) -> None:
        model = ActorCriticNetwork(
            input_dim=8,
            action_dim=4,
            hidden_sizes=[64, 64],
            activation="tanh",
        )
        observations = torch.randn(7, 8)

        output = model.act(observations)

        self.assertEqual(tuple(output.policy.logits.shape), (7, 4))
        self.assertEqual(tuple(output.policy.actions.shape), (7,))
        self.assertEqual(tuple(output.values.shape), (7,))


if __name__ == "__main__":
    unittest.main()