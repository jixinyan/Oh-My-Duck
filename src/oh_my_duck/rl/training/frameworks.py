"""RL framework selection independent of physics; safe to import without ML packages."""
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Callable, Sequence
from oh_my_duck.rl.training.base import BackendCommand
from oh_my_duck.rl.training.isaac_newton import is_diagnostic, require_task
from oh_my_duck.rl.training.registry import BackendUnavailable, default_registry

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
        logger_key, project_key = (("--agent.logger", "--agent.wandb-project") if backend == "mujoco" or not is_diagnostic(arguments)
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
    if backend == 'mujoco' or not is_diagnostic(arguments):
        if backend == 'isaac-newton' and operation == 'train':
            require_task(root, arguments)
        environment_name = 'mujoco-sb3' if backend == 'mujoco' or operation == 'export' else 'isaac-newton'
        interpreter = root / f'.envs/{environment_name}/bin/python'
        if not interpreter.exists():
            raise BackendUnavailable(f'Run python omd.py setup --backend {backend} --rl-framework sb3 first')
        extras = ['--backend', backend] if operation == 'train' else []
        environment = {'PYTHONPATH': str(root/'src'), 'OMD_PROJECT_ROOT': str(root),
                       'MUJOCO_GL': 'egl', 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'offline'}
        return BackendCommand((str(interpreter), '-m', ('oh_my_duck.rl.learners.sb3.train' if operation == 'train'
            else 'oh_my_duck.rl.artifacts.sb3_export'), *arguments, *extras), root, environment)
    command = default_registry().get(backend, root).command(operation, arguments)
    return replace(command, argv=(command.argv[0], "-m", "oh_my_duck.rl.backends.isaac_newton.sb3_train", *arguments))


def default_framework_registry():
    registry = RLFrameworkRegistry()
    registry.register("rsl-rl", "RSL-RL", [
        FrameworkBinding("mujoco", ("train", "export"), "representative_smoke_resume_export_passed", _rsl_rl),
        FrameworkBinding("isaac-newton", ("train", "export"), "representative_smoke_resume_export_passed", _rsl_rl,
                         "Registered BAM tasks use shared native PPO; PD diagnostic remains explicit"),
    ])
    registry.register("sb3", "Stable-Baselines3", [
        FrameworkBinding("mujoco", ("train", "export"), "representative_smoke_resume_export_passed", _sb3, "Official mjlab task; 64-env smoke, 768 timeouts, and normalized official-runner export passed; gait quality unvalidated"),
        FrameworkBinding("isaac-newton", ("train", "export"), "representative_smoke_resume_export_passed", _sb3,
                         "Registered BAM tasks use native SB3; normalized export uses official MuJoCo metadata reference"),
    ])
    return registry
