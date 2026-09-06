"""Online robot execution adapters: simulated and real, distinct from training backends."""
from oh_my_duck.robotics.backends.base import RobotBackend, RobotCapabilities, RobotState
__all__ = ["RobotBackend", "RobotCapabilities", "RobotState"]
