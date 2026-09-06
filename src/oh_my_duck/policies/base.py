from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class PolicyKind(StrEnum):
    JOINT = "joint_policy"
    VELOCITY = "velocity_policy"
    ACTION_SEQUENCE = "action_sequence_policy"


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    version: str
    kind: PolicyKind
    observation_schema: str
    action_schema: str
    update_hz: float
    artifact_ref: str
    sha256: str
    runtime: str


class PolicyAdapter(Protocol):
    @property
    def spec(self) -> PolicySpec: ...
    def reset(self) -> None: ...
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...
