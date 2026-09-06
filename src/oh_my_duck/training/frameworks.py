"""RL framework selection independent of physics; safe to import without ML packages."""
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Callable, Sequence
from .base import BackendCommand
from .registry import BackendUnavailable, default_registry

CommandFactory = Callable[[str, Path, str, Sequence[str]], BackendCommand]


@dataclass(frozen=True)
class FrameworkBinding:
    backend: str
    operations: tuple[str, ...]
    state: str
    factory: CommandFactory | None
    note: str = ""


class RLFrameworkRegistry:
    def __init__(self):
        self._frameworks: dict[str, tuple[str, tuple[FrameworkBinding, ...]]] = {}

    def register(self, name: str, title: str, bindings: Sequence[FrameworkBinding]):
        bindings = tuple(bindings)
        if not name or name in self._frameworks:
            raise ValueError(f"Empty or duplicate RL framework: {name!r}")
        if len({item.backend for item in bindings}) != len(bindings):
            raise ValueError(f"Duplicate backend binding for {name}")
        self._frameworks[name] = (title, bindings)

    def describe(self):
        return {name: {"title": title, "backends": {binding.backend: {
            "operations": list(binding.operations), "state": binding.state, "note": binding.note
        } for binding in bindings}} for name, (title, bindings) in self._frameworks.items()}

    def command(self, framework: str, backend: str, root: Path, operation: str,
                arguments: Sequence[str]) -> BackendCommand:
        if framework not in self._frameworks:
            raise BackendUnavailable(f"Unknown RL framework: {framework}; use omd frameworks")
        bindings = self._frameworks[framework][1]
        binding = next((item for item in bindings if item.backend == backend), None)
        if binding is None or binding.factory is None or operation not in binding.operations:
            note = binding.note if binding else "No adapter is registered"
            raise BackendUnavailable(f"{framework} / {backend} does not implement {operation}: {note}. No framework or physics fallback was selected.")
        return binding.factory(backend, root, operation, arguments)


def _rsl_rl(backend, root, operation, arguments):
    # Preserve the official backend-specific RSL-RL launch/export integrations.
    arguments = list(arguments)
    if operation == "train":
        defaults = json.loads((root / "configs/training.json").read_text())["logger"]
        options = {a.split("=", 1)[0] for a in arguments}
        logger_key, project_key = (("--agent.logger", "--agent.wandb-project") if backend == "mujoco"
                                   else ("--logger", "--log_project_name"))
        if logger_key not in options:
            arguments += [logger_key, "wandb"]
        if project_key not in options:
            arguments += [project_key, os.environ.get("WANDB_PROJECT", defaults["project"])]
    command = default_registry().get(backend, root).command(operation, arguments)
    if operation == "train":
        command = replace(command, environment={**command.environment, "WANDB_MODE": defaults["mode"]})
    return command


def _sb3(backend, root, operation, arguments):
    if backend == "mujoco":
        interpreter = root / ".envs/mujoco-sb3/bin/python"
        if not interpreter.exists():
            raise BackendUnavailable("Run python omd.py setup --backend mujoco --rl-framework sb3 first")
        return BackendCommand((str(interpreter), str(root / "training/mujoco" / ("sb3_train.py" if operation == "train" else "sb3_export.py")), *arguments),
                              root, {"MUJOCO_GL": "egl", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    command = default_registry().get(backend, root).command(operation, arguments)
    return replace(command, argv=(command.argv[0], "-m", "omd_isaac.sb3_train", *arguments))


def default_framework_registry():
    registry = RLFrameworkRegistry()
    registry.register("rsl-rl", "RSL-RL", [
        FrameworkBinding("mujoco", ("train", "export"), "official_walking_smoke_and_export_passed", _rsl_rl),
        FrameworkBinding("isaac-newton", ("train", "export"), "diagnostic_train_and_numerical_export_passed", _rsl_rl,
                         "Explicit PD diagnostic task only; BAM locomotion pending"),
    ])
    registry.register("sb3", "Stable-Baselines3", [
        FrameworkBinding("mujoco", ("train", "export"), "official_walking_train_resume_and_export_passed", _sb3, "Official mjlab task; 64-env smoke, 768 timeouts, and normalized official-runner export passed; gait quality unvalidated"),
        FrameworkBinding("isaac-newton", ("train",), "diagnostic_train_and_timeouts_passed", _sb3,
                         "Optional SB3 install; PD PPO only; native checkpoint resume and ONNX export pending"),
    ])
    return registry
