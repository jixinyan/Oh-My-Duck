"""Online robot execution adapters: simulated and real, distinct from training backends."""
from .base import RobotBackend, RobotCapabilities, RobotState
__all__ = ["RobotBackend", "RobotCapabilities", "RobotState"]
