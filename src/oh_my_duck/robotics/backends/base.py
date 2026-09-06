"""Execution interface; transport and physical state confirmation belong in concrete adapters."""
from dataclasses import dataclass
from typing import Protocol

from oh_my_duck.core.contracts.identity import Identity
from oh_my_duck.core.contracts.sensors import SensorFrame


@dataclass(frozen=True)
class RobotCapabilities:
    identity: Identity
    backend: str
    sensors: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    audio_capture: bool = False
    audio_playback: bool = False


@dataclass(frozen=True)
class RobotState:
    identity: Identity
    observed_at: str
    clock_domain: str
    motion_state: str
    active_task_id: str | None = None


@dataclass(frozen=True)
class StopResult:
    request_id: str
    robot_id: str
    confirmed_stopped: bool
    evidence_refs: tuple[str, ...]
    reason: str | None = None


class RobotBackend(Protocol):
    async def capabilities(self) -> RobotCapabilities: ...
    async def state(self) -> RobotState: ...
    async def read_sensor(self, sensor_id: str, *, max_age_ms: int) -> SensorFrame: ...
    async def command_velocity(
        self, vx_m_s: float, vy_m_s: float, yaw_rad_s: float, *, request_id: str, ttl_ms: int
    ) -> None: ...
    async def stop_motion(self, *, request_id: str) -> StopResult: ...
