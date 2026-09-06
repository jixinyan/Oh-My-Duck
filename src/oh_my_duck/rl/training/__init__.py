"""Offline training contracts; importing these interfaces does not import a simulator."""
from oh_my_duck.rl.training.base import TrainingBackend, TrainingRequest, PolicyArtifact
__all__ = ["TrainingBackend", "TrainingRequest", "PolicyArtifact"]
