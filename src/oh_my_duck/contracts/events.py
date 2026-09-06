"""Evidence schema shared by tools, voice and execution backends."""
from dataclasses import dataclass, field
from typing import Any

from .identity import Identity


@dataclass(frozen=True)
class EpisodeEvent:
    schema_version: int
    event_id: str
    episode_id: str
    session_id: str
    identity: Identity
    kind: str
    occurred_at: str
    clock_domain: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    task_id: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported episode schema version")
        if not all((self.event_id, self.episode_id, self.session_id, self.kind, self.clock_domain)):
            raise ValueError("Episode events require identity, kind and clock-domain fields")
