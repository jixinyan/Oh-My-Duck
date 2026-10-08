from dataclasses import asdict

from oh_my_duck.agentic.tools.catalog import ToolCatalog, ToolDefinition, ToolResult


def simulation_tools(backend, runner) -> ToolCatalog:
    catalog = ToolCatalog()

    async def capabilities(request_id, arguments):
        return ToolResult(request_id, "ok", asdict(await backend.capabilities()))

    async def state(request_id, arguments):
        return ToolResult(request_id, "ok", asdict(await backend.state()))

    async def sensor(request_id, arguments):
        return ToolResult(request_id, "ok", asdict(await backend.read_sensor(**arguments)))

    async def start(request_id, arguments):
        handle = await runner.start(arguments["skill_id"], arguments["parameters"], request_id=request_id)
        return ToolResult(request_id, "accepted", asdict(handle))

    async def status(request_id, arguments):
        handle = runner.handle(arguments["task_id"])
        return ToolResult(request_id, "ok", asdict(await runner.status(handle)))

    async def cancel(request_id, arguments):
        handle = runner.handle(arguments["task_id"])
        result = await runner.cancel(handle, reason=arguments.get("reason", "requested"))
        return ToolResult(request_id, "ok", asdict(result))

    async def stop(request_id, arguments):
        result = await runner.stop_active(request_id=request_id)
        return ToolResult(request_id, "ok", asdict(result))

    empty = {"type": "object", "additionalProperties": False}
    task = {"type": "string", "minLength": 1}
    for name, description, schema, handler in (
        ("get_capabilities", "Report actual robot execution capabilities", empty, capabilities),
        ("get_robot_state", "Read the latest measured robot state", empty, state),
        ("read_sensor", "Read a supported timestamped measurement", {
            "type": "object", "additionalProperties": False,
            "required": ["sensor_id", "max_age_ms"], "properties": {
                "sensor_id": {"type": "string", "minLength": 1},
                "max_age_ms": {"type": "integer", "minimum": 0},
            },
        }, sensor),
        ("run_skill", "Start a bounded velocity skill", {
            "type": "object", "additionalProperties": False,
            "required": ["skill_id", "parameters"], "properties": {
                "skill_id": {"type": "string", "const": "move_for"},
                "parameters": {
                    "type": "object", "additionalProperties": False,
                    "required": ["vx_m_s", "vy_m_s", "yaw_rad_s", "duration_s"],
                    "properties": {
                        "vx_m_s": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                        "vy_m_s": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                        "yaw_rad_s": {"type": "number", "minimum": -2, "maximum": 2},
                        "duration_s": {"type": "number", "minimum": 0.02, "maximum": 60},
                    },
                },
            },
        }, start),
        ("get_task_status", "Read a task result from this execution session", {
            "type": "object", "additionalProperties": False,
            "required": ["task_id"], "properties": {"task_id": task},
        }, status),
        ("cancel_task", "Cancel a task and confirm physical stop", {
            "type": "object", "additionalProperties": False,
            "required": ["task_id"], "properties": {
                "task_id": task, "reason": {"type": "string", "minLength": 1},
            },
        }, cancel),
        ("stop_motion", "Set zero command and confirm physical stop", empty, stop),
    ):
        catalog.register(ToolDefinition(name, description, schema), handler)
    return catalog
