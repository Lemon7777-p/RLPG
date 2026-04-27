from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_benchmark.algorithms import RLAlgorithm, resolve_device
from rl_benchmark.models import DiscretePolicyNetwork


class DummyPolicyAlgorithm(RLAlgorithm):
    def __init__(self, device: str = "auto") -> None:
        super().__init__(
            name="dummy-policy",
            config={"algorithm": {"name": "dummy-policy"}},
            device=device,
        )
        self.policy = DiscretePolicyNetwork(
            input_dim=4,
            action_dim=2,
            hidden_sizes=[16],
            activation="tanh",
        )
        self.finalize_setup()

    def act(self, observations: np.ndarray | torch.Tensor, deterministic: bool = False):
        observation_tensor = self.prepare_tensor(observations)
        return self.policy.act(observation_tensor, deterministic=deterministic)

    def update(self, batch):
        return {"loss": 0.0}


class ResolveDeviceTests(unittest.TestCase):
    def test_auto_device_resolves_to_available_backend(self) -> None:
        device = resolve_device("auto")

        self.assertIn(device.type, {"cpu", "cuda"})


class AlgorithmBaseTests(unittest.TestCase):
    def test_prepare_tensor_batches_single_observation(self) -> None:
        algorithm = DummyPolicyAlgorithm(device="cpu")

        tensor = algorithm.prepare_tensor([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(tuple(tensor.shape), (1, 4))
        self.assertEqual(tensor.device.type, "cpu")

    def test_act_returns_policy_output(self) -> None:
        algorithm = DummyPolicyAlgorithm(device="cpu")
        observations = np.ones((3, 4), dtype=np.float32)

        output = algorithm.act(observations, deterministic=True)

        self.assertEqual(tuple(output.actions.shape), (3,))
        self.assertEqual(tuple(output.log_probs.shape), (3,))

    def test_checkpoint_round_trip_restores_weights(self) -> None:
        algorithm = DummyPolicyAlgorithm(device="cpu")
        restored = DummyPolicyAlgorithm(device="cpu")

        with torch.no_grad():
            first_parameter = next(algorithm.parameters())
            first_parameter.fill_(0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "dummy.pt"
            algorithm.save_checkpoint(checkpoint_path, metadata={"epoch": 1})
            payload = restored.load_checkpoint(checkpoint_path)

        original_weights = next(algorithm.parameters())
        restored_weights = next(restored.parameters())
        self.assertTrue(torch.allclose(original_weights, restored_weights))
        self.assertEqual(payload["metadata"]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()