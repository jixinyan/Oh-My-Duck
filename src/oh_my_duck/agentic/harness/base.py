from dataclasses import dataclass
from typing import Literal, Protocol

from oh_my_duck.core.contracts.events import EpisodeEvent
from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.agentic.tools.catalog import ToolDefinition, ToolResult


HarnessStatus = Literal["completed", "failed", "unsupported", "cancelled"]


@dataclass(frozen=True)
class HarnessDispatch:
    """The bridge result for one user request or cancellation.

    A completed harness dispatch only reports the local tool result.  Physical
    success remains a task/verifier decision owned by the external harness.
    """

    request_id: str
    session_id: str
    status: HarnessStatus
    tool_name: str | None = None
    tool_result: ToolResult | None = None
    reason: str | None = None


class HarnessUnavailable(RuntimeError):
    """Raised by a future transport adapter when the external harness is absent."""


@dataclass(frozen=True)
class HarnessEndpoint:
    """Deployment metadata kept separate from the not-yet-stable wire protocol."""

    endpoint: str
    protocol: str = "edh"
    source_revision: str | None = None


class UnavailableHarnessBridge:
    """Visible failure boundary until an external harness transport is wired."""

    def __init__(self, endpoint: HarnessEndpoint):
        self.endpoint = endpoint

    def _raise(self) -> None:
        raise HarnessUnavailable(
            f"External harness transport is unavailable for {self.endpoint.endpoint!r}"
        )

    async def register_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        del tools
        self._raise()

    async def submit_text(
        self,
        text: str,
        *,
        session_id: str,
        request_id: str,
        audio: PayloadRef | None = None,
    ) -> HarnessDispatch:
        del text, session_id, request_id, audio
        self._raise()

    async def publish_event(self, event: EpisodeEvent) -> None:
        del event
        self._raise()

    async def submit_experience(self, *, episode_id: str, evidence_refs: tuple[str, ...]) -> None:
        del episode_id, evidence_refs
        self._raise()

    async def cancel(self, request_id: str, *, session_id: str, reason: str) -> HarnessDispatch:
        del request_id, session_id, reason
        self._raise()


class HarnessBridge(Protocol):
    async def register_tools(self, tools: tuple[ToolDefinition, ...]) -> None: ...
    async def submit_text(
        self,
        text: str,
        *,
        session_id: str,
        request_id: str,
        audio: PayloadRef | None = None,
    ) -> HarnessDispatch: ...
    async def publish_event(self, event: EpisodeEvent) -> None: ...
    async def submit_experience(self, *, episode_id: str, evidence_refs: tuple[str, ...]) -> None: ...
    async def cancel(self, request_id: str, *, session_id: str, reason: str) -> HarnessDispatch: ...
