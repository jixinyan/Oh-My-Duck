"""Persistent persona and physical/simulated robot identity are separate."""
from dataclasses import dataclass
from enum import StrEnum


class ExecutionDomain(StrEnum):
    SIMULATION = "simulation"
    REAL = "real"


@dataclass(frozen=True)
class Identity:
    persona_id: str
    robot_id: str
    domain: ExecutionDomain

    def __post_init__(self):
        if not self.persona_id or not self.robot_id:
            raise ValueError("persona_id and robot_id must be non-empty")
