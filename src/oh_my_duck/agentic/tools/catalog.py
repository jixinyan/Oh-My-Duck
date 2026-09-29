"""A small tool registry with boundary validation.

The external harness owns tool selection and dispatch policy.  This module only
keeps the local definition-to-handler binding and rejects malformed arguments
before a handler can affect a robot or persist an event.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
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
    """Raised when tool arguments do not satisfy the declared JSON schema."""

    def __init__(self, errors: tuple[str, ...]):
        self.errors = errors
        super().__init__("Invalid tool arguments: " + "; ".join(errors))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "array"
    return "unknown"


def _matches_type(value: Any, expected: str) -> bool:
    actual = _json_type(value)
    if expected == "number":
        return actual in {"integer", "number"}
    return actual == expected


def _check_schema(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal const")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of enum values")

    expected = schema.get("type")
    if expected is not None:
        types = (expected,) if isinstance(expected, str) else tuple(expected)
        if not any(_matches_type(value, item) for item in types):
            errors.append(f"{path}: expected type {expected!r}, got {_json_type(value)!r}")
            return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            errors.append(f"{path}: must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: must contain at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: must contain at most {schema['maxLength']} characters")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: must contain at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _check_schema(item, item_schema, f"{path}[{index}]", errors)
    if isinstance(value, Mapping):
        required = schema.get("required", ())
        for name in required:
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            child_path = f"{path}.{name}"
            if name in properties and isinstance(properties[name], Mapping):
                _check_schema(item, properties[name], child_path, errors)
            elif additional is False:
                errors.append(f"{child_path}: additional property is not allowed")
            elif isinstance(additional, Mapping):
                _check_schema(item, additional, child_path, errors)

    for keyword in ("allOf", "anyOf", "oneOf"):
        choices = schema.get(keyword)
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
            continue
        matched = 0
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            branch: list[str] = []
            _check_schema(value, choice, path, branch)
            matched += not branch
        if keyword == "allOf" and matched != len(choices):
            errors.append(f"{path}: allOf constraint failed")
        elif keyword == "anyOf" and matched == 0:
            errors.append(f"{path}: anyOf constraint failed")
        elif keyword == "oneOf" and matched != 1:
            errors.append(f"{path}: oneOf constraint failed")


def validate_tool_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the JSON-schema subset used by native harness tool inputs."""

    if not isinstance(arguments, Mapping):
        raise ToolSchemaError(("$: expected an object",))
    errors: list[str] = []
    _check_schema(arguments, schema, "$", errors)
    if errors:
        raise ToolSchemaError(tuple(errors))


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
        definition, handler = self._tools[name]
        validate_tool_arguments(arguments, definition.parameter_schema)
        result = await handler(request_id, arguments)
        if result.request_id != request_id:
            raise ValueError("Tool result does not match the request")
        return result
