"""Shared wire-level concepts. Simulator and model dependencies never belong here."""
from .identity import ExecutionDomain, Identity
from .tasks import TaskHandle, TaskStatus
from .sensors import SensorFrame
from .events import EpisodeEvent

__all__ = ["ExecutionDomain", "Identity", "TaskHandle", "TaskStatus", "SensorFrame", "EpisodeEvent"]
