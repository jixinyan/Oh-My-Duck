from dataclasses import asdict

from oh_my_duck.agentic.tools.catalog import ToolCatalog, ToolDefinition, ToolResult


def simulation_tools(backend, runner) -> ToolCatalog:
    catalog = ToolCatalog()

    async def capabilities(request_id, arguments):
        if arguments:
            raise ValueError("get_capabilities takes no arguments")
        return ToolResult(request_id, "ok", asdict(await backend.capabilities()))

    async def state(request_id, arguments):
        if arguments:
            raise ValueError("get_robot_state takes no arguments")
        return ToolResult(request_id, "ok", asdict(await backend.state()))

    async def sensor(request_id, arguments):
        if set(arguments) != {"sensor_id", "max_age_ms"}:
            raise ValueError("read_sensor requires sensor_id and max_age_ms")
        return ToolResult(request_id, "ok", asdict(await backend.read_sensor(**arguments)))

    async def start(request_id, arguments):
        if set(arguments) != {"skill_id", "parameters"} or not isinstance(arguments["parameters"], dict):
            raise ValueError("run_skill requires skill_id and parameters")
        handle = await runner.start(arguments["skill_id"], arguments["parameters"], request_id=request_id)
        return ToolResult(request_id, "accepted", asdict(handle))

    async def status(request_id, arguments):
        if set(arguments) != {"task_id"}:
            raise ValueError("get_task_status requires task_id")
        handle = runner.handle(arguments["task_id"])
        return ToolResult(request_id, "ok", asdict(await runner.status(handle)))

    async def cancel(request_id, arguments):
        if set(arguments) not in ({"task_id"}, {"task_id", "reason"}):
            raise ValueError("cancel_task requires task_id and optional reason")
        handle = runner.handle(arguments["task_id"])
        result = await runner.cancel(handle, reason=arguments.get("reason", "requested"))
        return ToolResult(request_id, "ok", asdict(result))

    async def stop(request_id, arguments):
        if arguments:
            raise ValueError("stop_motion takes no arguments")
        result = await runner.stop_active(request_id=request_id)
        return ToolResult(request_id, "ok", asdict(result))

    for name, description, handler in (
        ("get_capabilities", "Report actual robot execution capabilities", capabilities),
        ("get_robot_state", "Read the latest measured robot state", state),
        ("read_sensor", "Read a supported timestamped measurement", sensor),
        ("run_skill", "Start a bounded velocity skill", start),
        ("get_task_status", "Read a task result from this execution session", status),
        ("cancel_task", "Cancel a task and confirm physical stop", cancel),
        ("stop_motion", "Set zero command and confirm physical stop", stop),
    ):
        catalog.register(ToolDefinition(name, description, {"type": "object"}), handler)
    return catalog
