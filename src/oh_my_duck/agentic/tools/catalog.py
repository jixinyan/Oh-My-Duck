from collections.abc import Mapping
from dataclasses import dataclass
import json
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


class ToolSchemaError(ValueError):
    def __init__(self, errors: tuple[str, ...]):
        self.errors = errors
        super().__init__("Invalid tool arguments: " + "; ".join(errors))


def validate_tool_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    if not isinstance(arguments, Mapping):
        raise ToolSchemaError(("$: expected an object",))
    # 参数与 Schema 均遵守 JSON 数值规则。
    json.dumps(schema, allow_nan=False)
    Draft202012Validator.check_schema(schema)
    json.dumps(arguments, allow_nan=False)
    errors = tuple(
        f"{error.json_path}: {error.message}"
        for error in Draft202012Validator(schema).iter_errors(arguments)
    )
    if errors:
        raise ToolSchemaError(errors)


class ToolCatalog:
    def __init__(self):
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        from jsonschema import Draft202012Validator

        if not definition.name or definition.name in self._tools:
            raise ValueError(f"Empty or duplicate tool name: {definition.name!r}")
        json.dumps(definition.parameter_schema, allow_nan=False)
        Draft202012Validator.check_schema(definition.parameter_schema)
        self._tools[definition.name] = (definition, handler)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(definition for definition, _ in self._tools.values())

    async def invoke(self, name: str, request_id: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(request_id, "unsupported", {"tool": name})
        definition, handler = self._tools[name]
        validate_tool_arguments(arguments, definition.parameter_schema)
        result = await handler(request_id, arguments)
        if result.request_id != request_id:
            raise ValueError("Tool result does not match the request")
        return result
