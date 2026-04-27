"""Helpers for launching dashboard-triggered training in background processes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rl_benchmark import PROJECT_ROOT
from rl_benchmark.logging.schema import MANIFEST_FILENAME, RunManifest, read_manifest, run_dir_for, utc_now_iso
from rl_benchmark.runners import build_run_id


BACKGROUND_LAUNCH_FILENAME = "background_launch.json"
BACKGROUND_LOG_FILENAME = "background_train.log"


@dataclass(slots=True)
class BackgroundRunLaunch:
    run_id: str
    run_dir: Path
    log_path: Path
    launch_metadata_path: Path
    pid: int
    command: list[str]


@dataclass(slots=True)
class BackgroundRunRequest:
    algorithm_name: str
    env_id: str
    seed: int
    device: str
    train_steps: int
    eval_episodes: int
    results_root: Path
    notes: str
    checkpoint_interval_steps: int
    resume: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results_root"] = str(self.results_root)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackgroundRunRequest":
        return cls(
            algorithm_name=str(payload["algorithm_name"]),
            env_id=str(payload["env_id"]),
            seed=int(payload["seed"]),
            device=str(payload.get("device", "auto")),
            train_steps=int(payload["train_steps"]),
            eval_episodes=int(payload.get("eval_episodes", 1)),
            results_root=Path(payload["results_root"]),
            notes=str(payload.get("notes", "")),
            checkpoint_interval_steps=int(payload.get("checkpoint_interval_steps", 0)),
            resume=bool(payload.get("resume", False)),
        )

    def to_launch_kwargs(self) -> dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "env_id": self.env_id,
            "seed": self.seed,
            "device": self.device,
            "train_steps": self.train_steps,
            "eval_episodes": self.eval_episodes,
            "results_root": self.results_root,
            "notes": self.notes,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "resume": self.resume,
        }


@dataclass(slots=True)
class BackgroundRunInfo:
    run_id: str
    pid: int | None
    command: list[str]
    resume: bool
    launched_at: str | None
    log_path: Path
    launch_metadata_path: Path
    log_exists: bool
    log_size_bytes: int | None
    log_modified_at: str | None

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["log_path"] = str(self.log_path)
        payload["launch_metadata_path"] = str(self.launch_metadata_path)
        return payload


def launch_background_training_job(
    *,
    algorithm_name: str,
    env_id: str,
    seed: int,
    device: str,
    train_steps: int,
    eval_episodes: int,
    results_root: str | Path,
    notes: str,
    checkpoint_interval_steps: int,
    resume: bool,
) -> BackgroundRunLaunch:
    run_id = build_run_id(algorithm_name, env_id, seed)
    run_dir = run_dir_for(run_id, results_root)
    _ensure_run_is_launchable(run_dir, resume=resume)
    request = BackgroundRunRequest(
        algorithm_name=algorithm_name,
        env_id=env_id,
        seed=seed,
        device=device,
        train_steps=train_steps,
        eval_episodes=eval_episodes,
        results_root=Path(results_root),
        notes=notes,
        checkpoint_interval_steps=checkpoint_interval_steps,
        resume=resume,
    )

    log_path = run_dir / BACKGROUND_LOG_FILENAME
    command = _build_command(request)

    env = os.environ.copy()
    env["CV_SHOW"] = env.get("CV_SHOW", "1")
    env["PYTHONPATH"] = _compose_pythonpath(env.get("PYTHONPATH"))

    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=_creationflags(),
            start_new_session=not _is_windows(),
        )

    launch_metadata_path = run_dir / BACKGROUND_LAUNCH_FILENAME
    launched_at = utc_now_iso()
    launch_metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": process.pid,
                "command": command,
                "log_path": str(log_path),
                "resume": resume,
                "launched_at": launched_at,
                "request": request.to_payload(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return BackgroundRunLaunch(
        run_id=run_id,
        run_dir=run_dir,
        log_path=log_path,
        launch_metadata_path=launch_metadata_path,
        pid=process.pid,
        command=command,
    )


def relaunch_background_training_job(
    run_dir: str | Path,
    *,
    resume: bool,
) -> BackgroundRunLaunch:
    request = load_background_run_request(run_dir, resume=resume)
    if request is None:
        raise FileNotFoundError(f"No persisted launch request is available for run: {Path(run_dir).name}")
    return launch_background_training_job(**request.to_launch_kwargs())


def load_background_run_request(
    run_dir: str | Path,
    *,
    resume: bool | None = None,
) -> BackgroundRunRequest | None:
    run_dir = Path(run_dir)
    manifest = _load_manifest_if_present(run_dir)
    launch_metadata_path = run_dir / BACKGROUND_LAUNCH_FILENAME
    request: BackgroundRunRequest | None = None

    if launch_metadata_path.is_file():
        with launch_metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        request_payload = payload.get("request")
        if isinstance(request_payload, dict):
            request = BackgroundRunRequest.from_payload(request_payload)
        else:
            request = _request_from_legacy_payload(payload, run_dir=run_dir, manifest=manifest)

    if request is None and manifest is not None:
        request = _request_from_manifest(manifest, run_dir=run_dir)

    if request is None:
        return None

    if resume is not None:
        request = replace(request, resume=resume)
    return request


def load_background_run_info(run_dir: str | Path) -> BackgroundRunInfo | None:
    run_dir = Path(run_dir)
    launch_metadata_path = run_dir / BACKGROUND_LAUNCH_FILENAME
    if not launch_metadata_path.is_file():
        return None

    with launch_metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_log_path = payload.get("log_path")
    log_path = Path(raw_log_path) if raw_log_path else run_dir / BACKGROUND_LOG_FILENAME
    log_stat = log_path.stat() if log_path.is_file() else None
    return BackgroundRunInfo(
        run_id=str(payload.get("run_id") or run_dir.name),
        pid=int(payload["pid"]) if payload.get("pid") is not None else None,
        command=[str(part) for part in payload.get("command", [])],
        resume=bool(payload.get("resume", False)),
        launched_at=str(payload.get("launched_at")) if payload.get("launched_at") else None,
        log_path=log_path,
        launch_metadata_path=launch_metadata_path,
        log_exists=log_stat is not None,
        log_size_bytes=int(log_stat.st_size) if log_stat is not None else None,
        log_modified_at=(
            datetime.fromtimestamp(log_stat.st_mtime, tz=timezone.utc).isoformat()
            if log_stat is not None
            else None
        ),
    )


def read_background_log_tail(run_dir: str | Path, *, max_lines: int = 40) -> str | None:
    run_dir = Path(run_dir)
    log_path = run_dir / BACKGROUND_LOG_FILENAME
    if not log_path.is_file():
        return None

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _build_command(request: BackgroundRunRequest) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_run.py"),
        "--algorithm",
        request.algorithm_name,
        "--env",
        request.env_id,
        "--seed",
        str(request.seed),
        "--device",
        request.device,
        "--train-steps",
        str(request.train_steps),
        "--eval-episodes",
        str(request.eval_episodes),
        "--results-root",
        str(request.results_root),
        "--notes",
        request.notes,
        "--checkpoint-interval-steps",
        str(request.checkpoint_interval_steps),
    ]
    if request.resume:
        command.append("--resume")
    return command


def _ensure_run_is_launchable(run_dir: Path, *, resume: bool) -> None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return

    manifest = read_manifest(manifest_path)
    if manifest.status == "running":
        raise RuntimeError(f"Run {manifest.run_id} is already active. Wait for it to finish or resume it later.")
    if resume and not manifest.latest_checkpoint:
        raise FileNotFoundError(f"Run {manifest.run_id} has no checkpoint to resume from.")


def _load_manifest_if_present(run_dir: Path) -> RunManifest | None:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    return read_manifest(manifest_path)


def _request_from_manifest(manifest: RunManifest, *, run_dir: Path) -> BackgroundRunRequest | None:
    config_snapshot = manifest.config_snapshot or {}
    environment_config = config_snapshot.get("environment", {})
    evaluation_config = config_snapshot.get("evaluation", {})
    runtime_config = config_snapshot.get("runtime", {})

    train_steps = max(int(environment_config.get("train_steps", 1)), int(manifest.total_steps or 0), 1)
    eval_episodes = max(int(evaluation_config.get("episodes", 1)), 1)
    checkpoint_interval_steps = max(int(runtime_config.get("checkpoint_interval_steps", 0)), 0)
    device = str(runtime_config.get("device", "auto"))
    return BackgroundRunRequest(
        algorithm_name=manifest.algorithm_name,
        env_id=manifest.env_id,
        seed=manifest.seed,
        device=device,
        train_steps=train_steps,
        eval_episodes=eval_episodes,
        results_root=run_dir.parent,
        notes=manifest.notes,
        checkpoint_interval_steps=checkpoint_interval_steps,
        resume=False,
    )


def _request_from_legacy_payload(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    manifest: RunManifest | None,
) -> BackgroundRunRequest | None:
    fallback = _request_from_manifest(manifest, run_dir=run_dir) if manifest is not None else None
    command = [str(part) for part in payload.get("command", [])]

    algorithm_name = _command_option(command, "--algorithm") or (fallback.algorithm_name if fallback else None)
    env_id = _command_option(command, "--env") or (fallback.env_id if fallback else None)
    seed = _command_option(command, "--seed")
    train_steps = _command_option(command, "--train-steps")
    eval_episodes = _command_option(command, "--eval-episodes")
    device = _command_option(command, "--device") or (fallback.device if fallback else "auto")
    results_root = _command_option(command, "--results-root")
    notes = _command_option(command, "--notes")
    checkpoint_interval_steps = _command_option(command, "--checkpoint-interval-steps")

    if algorithm_name is None or env_id is None:
        return fallback

    return BackgroundRunRequest(
        algorithm_name=algorithm_name,
        env_id=env_id,
        seed=int(seed) if seed is not None else (fallback.seed if fallback else 0),
        device=device,
        train_steps=int(train_steps) if train_steps is not None else (fallback.train_steps if fallback else 1),
        eval_episodes=int(eval_episodes) if eval_episodes is not None else (fallback.eval_episodes if fallback else 1),
        results_root=Path(results_root) if results_root is not None else (fallback.results_root if fallback else run_dir.parent),
        notes=notes if notes is not None else (fallback.notes if fallback else ""),
        checkpoint_interval_steps=(
            int(checkpoint_interval_steps)
            if checkpoint_interval_steps is not None
            else (fallback.checkpoint_interval_steps if fallback else 0)
        ),
        resume="--resume" in command or bool(payload.get("resume", False)),
    )


def _command_option(command: list[str], flag: str) -> str | None:
    try:
        flag_index = command.index(flag)
    except ValueError:
        return None

    if flag_index + 1 >= len(command):
        return None
    return str(command[flag_index + 1])


def _compose_pythonpath(existing_pythonpath: str | None) -> str:
    source_path = str(PROJECT_ROOT / "src")
    if not existing_pythonpath:
        return source_path

    parts = existing_pythonpath.split(os.pathsep)
    if source_path in parts:
        return existing_pythonpath
    return os.pathsep.join([source_path, existing_pythonpath])


def _creationflags() -> int:
    if not _is_windows():
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def _is_windows() -> bool:
    return os.name == "nt"