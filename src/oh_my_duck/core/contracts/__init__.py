"""Shared wire-level concepts. Simulator and model dependencies never belong here."""
from oh_my_duck.core.contracts.identity import ExecutionDomain, Identity
from oh_my_duck.core.contracts.tasks import TaskHandle, TaskStatus
from oh_my_duck.core.contracts.sensors import SensorFrame
from oh_my_duck.core.contracts.events import EpisodeEvent

__all__ = ["ExecutionDomain", "Identity", "TaskHandle", "TaskStatus", "SensorFrame", "EpisodeEvent"]
