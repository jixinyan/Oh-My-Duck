"""Training process registry. A missing implementation never selects another simulator."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from oh_my_duck.rl.training.base import BackendCommand, TrainingBackend


class BackendUnavailable(RuntimeError):
    pass


class TrainingBackendRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[[Path], TrainingBackend] | None] = {}

    def register(self, name: str, factory: Callable[[Path], TrainingBackend] | None) -> None:
        if not name or name in self._factories:
            raise ValueError(f"Empty or duplicate training backend: {name!r}")
        self._factories[name] = factory

    def get(self, name: str, root: Path) -> TrainingBackend:
        if name not in self._factories:
            raise BackendUnavailable(f"Unknown training backend: {name}")
        factory = self._factories[name]
        if factory is None:
            raise BackendUnavailable(f"{name} is planned but not implemented; no alternate physics backend was selected")
        return factory(root)


class MujocoTrainingBackend:
    def __init__(self, root: Path):
        self.root = root

    def command(self, operation: str, arguments: Sequence[str]) -> BackendCommand:
        interpreter = self.root / ".envs/mujoco/bin/python"
        if not interpreter.exists():
            raise BackendUnavailable("Run python omd.py setup first")
        if operation == "train":
            entry = ("-m", "oh_my_duck.rl.learners.rsl_rl.train")
        elif operation == "export":
            entry = ("-m", "oh_my_duck.rl.artifacts.export")
        elif operation in ("probe", "eval"):
            entry = ("-m", "oh_my_duck.rl.backends.mujoco.probe" if operation == "probe" else "oh_my_duck.rl.evaluation.mujoco")
        else:
            raise ValueError(f"Unsupported MuJoCo operation: {operation}")
        return BackendCommand((str(interpreter), *entry, *arguments), self.root,
                              {"MUJOCO_GL": "egl", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})


def default_registry() -> TrainingBackendRegistry:
    registry = TrainingBackendRegistry()
    registry.register("mujoco", MujocoTrainingBackend)
    from oh_my_duck.rl.training.isaac_newton import IsaacNewtonTrainingBackend
    registry.register("isaac-newton", IsaacNewtonTrainingBackend)
    return registry
