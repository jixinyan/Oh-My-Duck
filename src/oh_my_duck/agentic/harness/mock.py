"""A deterministic protocol peer for development before EDH exposes a stable API.

The mock has no model, planner, scheduler or memory.  A caller supplies exact
text-to-tool routes, so tests exercise request identity, tool registration,
argument validation, event publication and cancellation semantics without
pretending that an autonomous harness is available.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oh_my_duck.agentic.harness.base import HarnessDispatch
from oh_my_duck.agentic.tools import ToolCatalog, ToolDefinition, ToolResult
from oh_my_duck.core.contracts.events import EpisodeEvent
from oh_my_duck.core.contracts.sensors import PayloadRef


@dataclass(frozen=True)
class TextRoute:
    tool_name: str
    arguments: dict[str, Any]


class DeterministicHarnessMock:
    """Map preconfigured text to one registered tool invocation."""

    def __init__(self, catalog: ToolCatalog, routes: Mapping[str, TextRoute] | None = None):
        self.catalog = catalog
        self._routes = {self._normalize(text): route for text, route in (routes or {}).items()}
        self._definitions: dict[str, ToolDefinition] = {}
        self._requests: dict[str, tuple[str, str, HarnessDispatch]] = {}
        self._events: dict[str, EpisodeEvent] = {}
        self._experiences: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = " ".join(text.split()).casefold()
        if not normalized:
            raise ValueError("Harness text must not be empty")
        return normalized

    async def register_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        for definition in tools:
            if definition.name in self._definitions:
                raise ValueError(f"Duplicate harness tool: {definition.name!r}")
        for definition in tools:
            self._definitions[definition.name] = definition

    async def submit_text(
        self,
        text: str,
        *,
        session_id: str,
        request_id: str,
        audio: PayloadRef | None = None,
    ) -> HarnessDispatch:
        del audio  # The future EDH adapter will carry this reference in its message envelope.
        if not session_id or not request_id:
            raise ValueError("session_id and request_id are required")
        normalized = self._normalize(text)
        prior = self._requests.get(request_id)
        if prior is not None:
            old_session, old_text, dispatch = prior
            if old_session != session_id or old_text != normalized:
                raise ValueError("request_id was reused with different request identity")
            return dispatch

        route = self._routes.get(normalized)
        if route is None:
            dispatch = HarnessDispatch(request_id, session_id, "unsupported", reason="no deterministic route")
            self._requests[request_id] = (session_id, normalized, dispatch)
            return dispatch
        if route.tool_name not in self._definitions:
            dispatch = HarnessDispatch(
                request_id, session_id, "unsupported", route.tool_name, reason="tool is not registered"
            )
            self._requests[request_id] = (session_id, normalized, dispatch)
            return dispatch

        try:
            result = await self.catalog.invoke(route.tool_name, request_id, dict(route.arguments))
        except Exception as error:
            dispatch = HarnessDispatch(
                request_id, session_id, "failed", route.tool_name, reason=str(error)
            )
        else:
            status = (
                result.status
                if result.status in {"completed", "failed", "unsupported", "cancelled"}
                else "completed"
            )
            dispatch = HarnessDispatch(request_id, session_id, status, route.tool_name, result)
        self._requests[request_id] = (session_id, normalized, dispatch)
        return dispatch

    async def publish_event(self, event: EpisodeEvent) -> None:
        previous = self._events.get(event.event_id)
        if previous is not None and previous != event:
            raise ValueError("event_id was reused with different event data")
        self._events[event.event_id] = event

    async def submit_experience(self, *, episode_id: str, evidence_refs: tuple[str, ...]) -> None:
        if not episode_id:
            raise ValueError("episode_id is required")
        refs = tuple(dict.fromkeys(evidence_refs))
        previous = self._experiences.get(episode_id)
        if previous is not None and previous != refs:
            raise ValueError("episode_id was reused with different evidence")
        self._experiences[episode_id] = refs

    async def cancel(self, request_id: str, *, session_id: str, reason: str) -> HarnessDispatch:
        if not reason.strip():
            raise ValueError("Cancellation reason must not be empty")
        prior = self._requests.get(request_id)
        if prior is None or prior[0] != session_id:
            return HarnessDispatch(request_id, session_id, "unsupported", reason="unknown request")
        _, _, dispatch = prior
        if dispatch.status in {"completed", "failed", "unsupported", "cancelled"}:
            return dispatch
        cancelled = HarnessDispatch(request_id, session_id, "cancelled", dispatch.tool_name, reason=reason)
        self._requests[request_id] = (session_id, prior[1], cancelled)
        return cancelled

    @property
    def events(self) -> tuple[EpisodeEvent, ...]:
        return tuple(self._events.values())

    @property
    def experiences(self) -> dict[str, tuple[str, ...]]:
        return dict(self._experiences)
