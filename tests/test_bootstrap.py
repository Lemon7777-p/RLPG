from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_pyproject_is_parseable(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            data = tomllib.load(handle)

        self.assertEqual(data["project"]["name"], "rl-benchmark")
        self.assertIn("dependencies", data["project"])

    def test_expected_config_files_exist(self) -> None:
        expected_files = [
            ROOT / "configs" / "defaults.yaml",
            ROOT / "configs" / "environments.yaml",
            ROOT / "configs" / "algorithms" / "reinforce.yaml",
            ROOT / "configs" / "algorithms" / "a2c.yaml",
            ROOT / "configs" / "algorithms" / "ppo.yaml",
        ]

        for path in expected_files:
            self.assertTrue(path.is_file(), f"Missing config file: {path}")

    def test_package_bootstrap_import(self) -> None:
        source_root = ROOT / "src"
        self.assertTrue(source_root.is_dir())


if __name__ == "__main__":
    unittest.main()
