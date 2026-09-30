from __future__ import annotations

import asyncio
import json
from pathlib import Path
import secrets
from uuid import uuid4

from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    revision = "8a5e685b22d032207f53db20454f0992a4ad60fd"
    policy_revision = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"
    native_task_id = "official-apartment-office"
    secret = secrets.token_hex(32)
    emitted: list[dict] = []

    async def emit(event: dict) -> None:
        emitted.append(event)

    session = MicroDuckWorkerSession(emit)
    configuration = {
        "native_task_id": native_task_id,
        "catalog_dir": str(root / ".cache/official-policies" / policy_revision),
        "seed": 20260929,
        "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
        "task_instruction": "Navigate to the office",
        "policy_revision": policy_revision,
        "control_port": 0,
        "control_secret": secret,
    }
    await session.initialize({
        "provider": "microduck",
        "native_task_id": native_task_id,
        "execution_mode": "policy",
        "policy_id": "official-microduck-onnx",
        "scene_configuration": configuration,
        "schema_path": str(root / ".cache/edh" / revision /
                           "harness/contracts/schema/physical.schema.json"),
        "monitor_every_actions": 1,
        "policy_max_actions_per_inference": 1,
    })
    port = session._control_server.sockets[0].getsockname()[1]

    async def control(run_task_id: str, operation: str, arguments: dict) -> dict:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write((json.dumps({
                "run_task_id": run_task_id,
                "control_secret": secret,
                "operation": operation,
                "request_id": uuid4().hex,
                "arguments": arguments,
            }) + "\n").encode())
            await writer.drain()
            return json.loads(await reader.readline())
        finally:
            writer.close()
            await writer.wait_closed()

    first_run = str(uuid4())
    second_run = str(uuid4())
    third_run = str(uuid4())
    await session.open_task({"native_task_id": native_task_id,
                             "catalog_task_id": "navigate-office", "run_task_id": first_run})
    first = await control(first_run, "progress", {})
    if first["result"]["sequence"] != 0:
        raise AssertionError("First physical task did not start at the reset boundary")
    await session.start({"native_task_id": native_task_id, "request": {
        "schema_version": "physical.subgoal.v1",
        "task_id": first_run,
        "team_run_id": first_run,
        "decision_owner_id": str(uuid4()),
        "owner_assignment_id": str(uuid4()),
        "idempotency_key": str(uuid4()),
        "goal_id": "office-reached",
        "attempt_id": "attempt-1",
        "instruction": "Run the official velstand policy in the real apartment",
        "entities": {"robot": "microduck", "destination": "office"},
        "required_capabilities": ["policy-navigation"],
        "success_contract": {
            "id": "microduck-office-native", "version": "1",
            "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
            "source": {"kind": "benchmark", "reference": "official-microduck-apartment-native-gt"},
        },
        "budget": {"max_control_steps": 100, "max_wall_time_s": 30},
        "context_refs": [],
    }})
    await asyncio.sleep(0.6)
    stopped = await session.pause({}, terminal=False)
    boundary = stopped["status"]
    if boundary["control_steps"] < 5 or not boundary["device_confirmed"]:
        raise AssertionError("Real policy did not reach the stopped selection boundary")

    backend = session._environment._backend()
    def capture_many() -> None:
        for _ in range(8):
            backend.capture_observer()

    observing = asyncio.create_task(session._device.on_owner(capture_many))
    await asyncio.sleep(0.01)
    pending_selection = asyncio.create_task(control(
        first_run, "select_policy", {"policy_name": "alpha_walking"}))
    await asyncio.sleep(0.005)
    if pending_selection.done():
        raise AssertionError("Policy selection did not overlap real owner activity")
    await session.close_task()
    selected = await pending_selection
    await observing
    if selected.get("error", {}).get("message") != "MicroDuck tool request lacks the active task lease":
        raise AssertionError("Queued policy selection survived task closure")
    await session.open_task({"native_task_id": native_task_id,
                             "catalog_task_id": "navigate-office", "run_task_id": second_run})
    before = await control(second_run, "progress", {})
    stale = await control(first_run, "set_command", {"command": {"twist": [0.4, 0, 0]}})
    if stale.get("error", {}).get("message") != "MicroDuck tool request lacks the active task lease":
        raise AssertionError("Closed task command was not rejected at the worker boundary")

    observing = asyncio.create_task(session._device.on_owner(capture_many))
    await asyncio.sleep(0.01)
    pending_command = asyncio.create_task(control(
        second_run, "set_command", {"command": {"twist": [0.4, 0, 0]}}))
    await asyncio.sleep(0.005)
    if pending_command.done():
        raise AssertionError("Policy command did not overlap real owner activity")
    await session.close_task()
    commanded = await pending_command
    await observing
    if commanded.get("error", {}).get("message") != "MicroDuck tool request lacks the active task lease":
        raise AssertionError("Queued policy command survived task closure")
    await session.open_task({"native_task_id": native_task_id,
                             "catalog_task_id": "navigate-office", "run_task_id": third_run})
    after = await control(third_run, "progress", {})
    if before["result"]["episode_id"] != after["result"]["episode_id"] or (
        before["result"]["sequence"], before["result"]["command_block"]
    ) != (after["result"]["sequence"], after["result"]["command_block"]):
        raise AssertionError("Closed task command changed the retained physical scene")
    await session.close()
    print(json.dumps({"first_run": first_run, "second_run": second_run,
                      "third_run": third_run,
                      "episode_id": after["result"]["episode_id"],
                      "sequence": after["result"]["sequence"],
                      "control_steps_before_stop": boundary["control_steps"],
                      "queued_selection_rejected": True,
                      "queued_command_rejected": True,
                      "stale_command_rejected": True,
                      "retained_command_unchanged": True,
                      "native_events": len(emitted)}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
