"""Simulator-independent task catalog and lazy configuration construction.

The core imports no simulation or learning libraries. A task can add bindings
for new backends without changing the command dispatcher or learner code.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigRef:
    symbol: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        module, separator, name = self.symbol.partition(":")
        if not separator or not module or not name:
            raise ValueError(f"Expected module:symbol, got {self.symbol!r}")
        value = importlib.import_module(module)
        for part in name.split("."):
            value = getattr(value, part)
        return value

    def build(self, **kwargs):
        value = self.resolve()
        parameters = {**deepcopy(self.kwargs), **kwargs}
        if callable(value):
            return value(**parameters)
        if parameters:
            raise ValueError(f"Configuration object {self.symbol} takes no factory arguments")
        return deepcopy(value)


@dataclass(frozen=True)
class TaskBinding:
    environment: ConfigRef
    rsl_config: ConfigRef
    runner: ConfigRef
    robot_variant: ConfigRef | None = None
    runtime: ConfigRef | None = None


@dataclass(frozen=True)
class TaskSpec:
    id: str
    model: str
    bindings: dict[str, TaskBinding]
    representative: bool = False
    policy_configs: dict[str, ConfigRef] = field(default_factory=dict)
    evaluation: ConfigRef | None = None
    policy_package: ConfigRef | None = None

    def binding(self, backend: str) -> TaskBinding:
        try:
            return self.bindings[backend]
        except KeyError:
            raise ValueError(f"Task {self.id!r} has no {backend!r} binding") from None


class TaskRegistry:
    def __init__(self):
        self._tasks: dict[str, TaskSpec] = {}

    def register(self, task: TaskSpec):
        if not task.id or task.id in self._tasks or not task.bindings:
            raise ValueError(f"Empty, duplicate or unbound task: {task.id!r}")
        self._tasks[task.id] = deepcopy(task)

    def get(self, task_id: str) -> TaskSpec:
        try:
            return deepcopy(self._tasks[task_id])
        except KeyError:
            raise ValueError(f"Unknown project task: {task_id!r}") from None

    def list(self, backend: str | None = None) -> list[TaskSpec]:
        return [self.get(key) for key in sorted(self._tasks)
                if backend is None or backend in self._tasks[key].bindings]

    @classmethod
    def from_file(cls, path: Path):
        config = json.loads(path.read_text())
        if config["schema_version"] != 1:
            raise ValueError("Unsupported task registry schema")
        registry = cls()
        for task in config["tasks"]:
            bindings = {name: TaskBinding(**{key: ConfigRef(**value) if value is not None else None for key, value in values.items()})
                        for name, values in task["bindings"].items()}
            policies = {name: ConfigRef(**value) for name, value in task.get("policy_configs", {}).items()}
            registry.register(TaskSpec(**{**task, "bindings": bindings, "policy_configs": policies,
                "evaluation": ConfigRef(**task["evaluation"]) if task.get("evaluation") else None,
                "policy_package": ConfigRef(**task["policy_package"]) if task.get("policy_package") else None}))
        return registry


def project_tasks(root: Path | None = None) -> TaskRegistry:
    if root is None:
        import os
        from oh_my_duck.core.paths import project_root
        root = project_root()
    return TaskRegistry.from_file(root / "configs/tasks.json")
