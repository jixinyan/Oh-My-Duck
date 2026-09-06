"""A small tool registry; adapters own argument validation against their declared schema."""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameter_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    status: str
    payload: dict[str, Any]


ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


class ToolCatalog:
    def __init__(self):
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if not definition.name or definition.name in self._tools:
            raise ValueError(f"Empty or duplicate tool name: {definition.name!r}")
        self._tools[definition.name] = (definition, handler)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(definition for definition, _ in self._tools.values())

    async def invoke(self, name: str, request_id: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(request_id, "unsupported", {"tool": name})
        _, handler = self._tools[name]
        result = await handler(request_id, arguments)
        if result.request_id != request_id:
            raise ValueError("Tool result does not match the request")
        return result
