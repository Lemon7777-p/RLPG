"""Config loading helpers for experiment definitions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rl_benchmark import PROJECT_ROOT


CONFIG_ROOT = PROJECT_ROOT / "configs"
ALGORITHM_CONFIG_ROOT = CONFIG_ROOT / "algorithms"


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")

    return data


def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
            continue

        merged[key] = deepcopy(value)

    return merged


def load_defaults() -> dict[str, Any]:
    return load_yaml_file(CONFIG_ROOT / "defaults.yaml")


def load_environment_registry() -> dict[str, Any]:
    return load_yaml_file(CONFIG_ROOT / "environments.yaml")


def load_algorithm_config(name: str) -> dict[str, Any]:
    normalized_name = name.strip().lower()
    return load_yaml_file(ALGORITHM_CONFIG_ROOT / f"{normalized_name}.yaml")


def load_environment_config(env_id: str) -> dict[str, Any]:
    registry = load_environment_registry()
    environments = registry.get("environments", {})

    if env_id not in environments:
        raise KeyError(f"Unknown environment id: {env_id}")

    environment_config = environments[env_id]
    if not isinstance(environment_config, dict):
        raise ValueError(f"Environment entry must be a mapping: {env_id}")

    return deepcopy(environment_config)


def load_run_config(algorithm_name: str, env_id: str) -> dict[str, Any]:
    defaults = load_defaults()
    algorithm_config = load_algorithm_config(algorithm_name)
    environment_config = load_environment_config(env_id)

    run_config = merge_dicts(defaults, {"algorithm": algorithm_config["algorithm"]})
    run_config["environment"] = environment_config
    run_config["run"] = {
        "algorithm_name": algorithm_config["algorithm"]["name"],
        "env_id": env_id,
        "runtime_env_id": environment_config.get("runtime_env_id", env_id),
    }
    return run_config


def list_algorithms() -> list[str]:
    return sorted(path.stem for path in ALGORITHM_CONFIG_ROOT.glob("*.yaml"))


def list_environments() -> list[str]:
    registry = load_environment_registry()
    environments = registry.get("environments", {})
    return sorted(environments.keys())
