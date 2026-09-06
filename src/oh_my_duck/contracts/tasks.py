"""Robot execution results; the external harness owns general task scheduling."""
from dataclasses import dataclass
from enum import StrEnum


class TaskStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskHandle:
    task_id: str
    request_id: str
    robot_id: str


@dataclass(frozen=True)
class TaskResult:
    handle: TaskHandle
    status: TaskStatus
    evidence_refs: tuple[str, ...] = ()
    reason: str | None = None
