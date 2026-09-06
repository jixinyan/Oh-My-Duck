"""Frames carry measurement validity and a clock domain; unknown is never free space."""
from dataclasses import dataclass
from enum import StrEnum


class Validity(StrEnum):
    VALID = "valid"
    NO_TARGET = "no_target"
    INVALID = "invalid"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PayloadRef:
    uri: str
    media_type: str
    sha256: str | None = None


@dataclass(frozen=True)
class SensorFrame:
    sensor_id: str
    robot_id: str
    episode_id: str
    sequence: int
    capture_time: float
    clock_domain: str
    received_at: str
    frame_id: str
    validity: Validity
    payload: PayloadRef | None
    calibration_revision: str | None = None
