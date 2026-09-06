from dataclasses import dataclass
from typing import Protocol

from oh_my_duck.core.contracts.sensors import SensorFrame, Validity


@dataclass(frozen=True)
class LookTarget:
    frame_id: str
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class TargetRecord:
    target_id: str
    robot_id: str
    label: str
    validity: Validity
    observed_at: str
    clock_domain: str
    evidence_refs: tuple[str, ...]
    position: LookTarget | None = None


class ActivePerception(Protocol):
    async def look_at(self, target: LookTarget, *, request_id: str) -> None: ...
    async def inspect(self, direction: str, *, max_age_ms: int, request_id: str) -> tuple[SensorFrame, ...]: ...
    async def target(self, target_id: str, *, max_age_ms: int) -> TargetRecord: ...
