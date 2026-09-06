"""Offline process contracts, separate from online RobotBackend."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence


TrainingBackendName = Literal["mujoco", "isaac-newton"]


@dataclass(frozen=True)
class TrainingRequest:
    backend: TrainingBackendName
    task_id: str
    config_path: Path
    output_dir: Path
    seed: int
    rl_framework: str = "rsl-rl"


@dataclass(frozen=True)
class PolicyArtifact:
    policy_path: Path
    manifest_path: Path
    training_backend: TrainingBackendName
    validation: Literal["unvalidated", "sim_validated", "hardware_validated"]
    rl_framework: str = "rsl-rl"


@dataclass(frozen=True)
class BackendCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]


class TrainingBackend(Protocol):
    def command(self, operation: str, arguments: Sequence[str]) -> BackendCommand: ...
