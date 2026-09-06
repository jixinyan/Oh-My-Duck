from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts.tasks import TaskHandle, TaskResult


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    version: str
    description: str
    parameter_schema: dict[str, Any]
    required_sensors: tuple[str, ...]
    resources: tuple[str, ...]
    supported_backends: tuple[str, ...]
    success_condition: str
    cancellation_condition: str
    policy_ref: str | None = None


class SkillRunner(Protocol):
    async def start(self, skill_id: str, parameters: dict[str, Any], *, request_id: str) -> TaskHandle: ...
    async def status(self, handle: TaskHandle) -> TaskResult: ...
    async def cancel(self, handle: TaskHandle, *, reason: str) -> TaskResult: ...
