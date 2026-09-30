from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import secrets
import socket
from uuid import uuid4

from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def control(port: int, secret: str, run_task_id: str, operation: str,
                  arguments: dict) -> dict:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        request = {"run_task_id": run_task_id, "control_secret": secret,
                   "operation": operation, "request_id": uuid4().hex,
                   "arguments": arguments}
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        response = json.loads(await reader.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]
    finally:
        writer.close()
        await writer.wait_closed()


async def expect_control_error(port: int, secret: str, run_task_id: str,
                               operation: str, arguments: dict,
                               message: str) -> None:
    try:
        await control(port, secret, run_task_id, operation, arguments)
    except RuntimeError as error:
        if message not in str(error):
            raise
    else:
        raise AssertionError(f"Native control accepted a request requiring {message}")


async def wait_boundary(session: MicroDuckWorkerSession) -> dict:
    pump = session._pump
    if pump is None:
        raise RuntimeError("Native policy pump was not started")
    async with asyncio.timeout(180):
        await pump
        await session._await_motion_cleanup()
    status = session._status
    if status["state"] != "paused" or not status["device_confirmed"]:
        raise AssertionError("Motion guard did not confirm a native paused boundary")
    return status


async def run(root: Path, twist: tuple[float, float, float],
              expected_stop: str, segment_limit: int,
              warmup_control_steps: int, stabilize_control_steps: int,
              require_goal: bool, disconnect_on_guard: bool,
              test_recovery_lease: bool, test_finish_boundary: bool) -> dict:
    edh = root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd"
    catalog = root / ".cache/official-policies" / OFFICIAL_REVISION
    port = available_port()
    secret = secrets.token_hex(32)
    run_task_id = f"guard-{uuid4().hex}"
    events = []
    disconnect_task: asyncio.Task[None] | None = None

    async def emit(message: dict) -> None:
        nonlocal disconnect_task
        if message["event"] == "update":
            events.append({"event": "update", "state": message["data"]["status"]["state"],
                           "control_steps": message["data"]["status"]["control_steps"],
                           "motion_guard": message["data"].get("control", {}).get("motion_guard")})
            if (disconnect_on_guard and disconnect_task is None and
                    message["data"]["status"]["state"] == "pausing" and
                    message["data"].get("control", {}).get("motion_guard")):
                disconnect_task = asyncio.create_task(session.transport_disconnected())

    session = MicroDuckWorkerSession(emit)
    configuration = {
        "native_task_id": "official-apartment-office", "catalog_dir": str(catalog),
        "seed": 20260929, "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
        "task_instruction": "Navigate from the corridor to the office",
        "policy_revision": OFFICIAL_REVISION, "control_port": port,
        "control_secret": secret,
    }
    initialized = False
    try:
        await session.initialize({
            "provider": "microduck", "scene_configuration": configuration,
            "native_task_id": "official-apartment-office", "policy_id": "official-microduck-onnx",
            "execution_mode": "policy", "schema_path": str(edh / "harness/contracts/schema/physical.schema.json"),
            "monitor_every_actions": 1, "policy_max_actions_per_inference": 1,
        })
        initialized = True
        await session.open_task({"native_task_id": "official-apartment-office",
                                 "run_task_id": run_task_id,
                                 "catalog_task_id": "navigate-office"})
        request = {
            "schema_version": "physical.subgoal.v1", "task_id": run_task_id,
            "team_run_id": run_task_id, "goal_id": "office-reached", "attempt_id": "attempt-1",
            "instruction": "Inspect and control the official apartment",
            "entities": {"robot": "microduck", "destination": "office"},
            "required_capabilities": ["policy-navigation"],
            "success_contract": {"id": "microduck-office-native", "version": "1",
                                 "all": [{"check_id": "goal_reached", "check": "native_goal_reached",
                                          "args": []}],
                                 "source": {"kind": "benchmark", "reference": "official-apartment-native-gt"}},
            "budget": {"max_control_steps": 1000, "max_wall_time_s": 600},
            "context_refs": [], "decision_owner_id": "planner",
            "owner_assignment_id": "motion-guard-acceptance",
            "idempotency_key": uuid4().hex,
        }
        await session.start({"request": request, "native_task_id": "official-apartment-office",
                             "observation_ttl_s": 30, "device_timeout_s": 120,
                             "policy_timeout_s": 30})
        if disconnect_on_guard:
            pump = session._pump
            async with asyncio.timeout(180):
                await pump
                await session._await_motion_cleanup()
                if disconnect_task is None:
                    raise AssertionError("Physical guard never reached the stop hook")
                await disconnect_task
            stopped = session._device.executed_actions
            await asyncio.sleep(0.05)
            snapshot = session._gate.snapshot()
            if (snapshot["state"] != "ended" or not snapshot["device_confirmed"] or
                    session._device.executed_actions != stopped or session._host_connected):
                raise AssertionError("Disconnected guard changed the confirmed physical boundary")
            return {"run_task_id": run_task_id, "disconnect_boundary": snapshot,
                    "control_steps": stopped, "updates": events}
        first = await wait_boundary(session)
        if first["control_steps"] != session.DEFAULT_COMMAND_STEPS or (
            session._motion_guard["reason"] != "command_segment_complete"
        ):
            raise AssertionError("Default velstand segment did not stop at its physical bound")
        warmup_remaining = warmup_control_steps - first["control_steps"]
        while warmup_remaining:
            count = min(100, warmup_remaining)
            if count < 5:
                raise ValueError("Warmup remainder must contain at least five controls")
            await control(port, secret, run_task_id, "set_command",
                          {"command": {"twist": [0.0, 0.0, 0.0]},
                           "max_control_steps": count})
            prior = session._status
            await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                                  "boundary_id": prior["boundary_event_id"],
                                  "state_version": prior["state_version"]})
            status = await wait_boundary(session)
            if status["control_steps"] != warmup_control_steps - warmup_remaining + count:
                raise AssertionError("Velstand warmup did not use actual completed controls")
            warmup_remaining -= count
        selected = await control(port, secret, run_task_id, "select_policy",
                                 {"policy_name": "alpha_walking"})
        if selected["policy_name"] != "alpha_walking":
            raise AssertionError("Official walking policy was not selected")
        boundaries = []
        for _ in range(segment_limit):
            await control(port, secret, run_task_id, "set_command",
                          {"command": {"twist": list(twist)},
                           "max_control_steps": 100})
            prior = session._status
            await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                                  "boundary_id": prior["boundary_event_id"],
                                  "state_version": prior["state_version"]})
            status = await wait_boundary(session)
            progress = await control(port, secret, run_task_id, "progress", {})
            boundary = {"control_steps": status["control_steps"],
                        "reason": progress["motion_guard"]["reason"],
                        "sequence": progress["sequence"],
                        "body_position_m": progress["body_position_m"],
                        "contact_evidence": progress["contact_evidence"],
                        "central_tof_closest_mm": progress["motion_guard"]["central_tof_closest_mm"]}
            boundaries.append(boundary)
            if progress["contact_evidence"]["non_ground_external_contact_samples_total"]:
                raise AssertionError("Motion continued into an external obstacle")
            if boundary["reason"] == expected_stop and expected_stop != "command_segment_complete":
                break
        if not boundaries or boundaries[-1]["reason"] != expected_stop:
            raise AssertionError(f"Real native motion reached {boundaries}")
        finish_rejections = []
        if test_finish_boundary:
            boundary = session._gate.snapshot()
            finish_arguments = {key: boundary[key] for key in
                                ("execution_id", "generation", "boundary_id")}
            await expect_control_error(port, secret, run_task_id, "finish_policy",
                                       finish_arguments, "zero-twist command segment")
            finish_rejections.append("moving_command")
        recovery_lease = None
        if test_recovery_lease:
            repeated = {"command": {"twist": list(twist)}, "max_control_steps": 100}
            changed_request = {"command": {"twist": [-0.2, 0.3, 0.0]},
                               "max_control_steps": 100}
            await expect_control_error(port, secret, run_task_id, "set_command", repeated,
                                       "changed twist command")
            await expect_control_error(port, secret, run_task_id, "set_command",
                                       changed_request, "fresh ToF")
            tof = await control(port, secret, run_task_id, "read_sensor", {"sensor": "tof"})
            if tof["sequence"] != boundaries[-1]["sequence"]:
                raise AssertionError("Fresh ToF differs from the confirmed stopped sequence")
            await expect_control_error(port, secret, run_task_id, "set_command", repeated,
                                       "changed twist command")
            changed = await control(port, secret, run_task_id, "set_command",
                                    changed_request)
            prior = session._status
            await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                                  "boundary_id": prior["boundary_event_id"],
                                  "state_version": prior["state_version"]})
            resumed = await wait_boundary(session)
            recovered = await control(port, secret, run_task_id, "progress", {})
            if (resumed["control_steps"] <= boundaries[-1]["control_steps"] or
                    recovered["contact_evidence"]["non_ground_external_contact_samples_total"]):
                raise AssertionError("Changed recovery command lacked safe physical motion")
            recovery_lease = {"tof_sequence": tof["sequence"],
                              "changed_command": changed,
                              "resumed_control_steps": resumed["control_steps"],
                              "resumed_position_m": recovered["body_position_m"],
                              "resumed_guard": recovered["motion_guard"]}
        stabilization = None
        if stabilize_control_steps:
            await control(port, secret, run_task_id, "set_command",
                          {"command": {"twist": [0.0, 0.0, 0.0]},
                           "max_control_steps": stabilize_control_steps})
            if test_finish_boundary:
                boundary = session._gate.snapshot()
                finish_arguments = {key: boundary[key] for key in
                                    ("execution_id", "generation", "boundary_id")}
                await expect_control_error(port, secret, run_task_id, "finish_policy",
                                           finish_arguments, "five executed zero-command steps")
                finish_rejections.append("unexecuted_zero_command")
            prior = session._status
            await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                                  "boundary_id": prior["boundary_event_id"],
                                  "state_version": prior["state_version"]})
            status = await wait_boundary(session)
            stabilization = await control(port, secret, run_task_id, "progress", {})
            if (status["control_steps"] - boundaries[-1]["control_steps"] !=
                    stabilize_control_steps or stabilization["contact_evidence"][
                        "non_ground_external_contact_samples_total"]):
                raise AssertionError("Physical zero-command stabilization was incomplete")
        finish_result = None
        if test_finish_boundary:
            boundary = session._gate.snapshot()
            finish_arguments = {key: boundary[key] for key in
                                ("execution_id", "generation", "boundary_id")}
            finish_result = await control(port, secret, run_task_id, "finish_policy",
                                          finish_arguments)
            if (finish_result["execution"]["state"] != "ended" or
                    finish_result["stop_confirmation"]["zero_control_steps"] < 5 or
                    finish_result["stop_confirmation"]["stopped_samples"] < 5):
                raise AssertionError("Native finish lacked measured physical stop")
        native_check = await session._device.on_owner(
            session._environment._backend().check_goal)
        if require_goal and not native_check["checks"]["goal_reached"]["satisfied"]:
            raise AssertionError(f"Native goal remains unsatisfied: {native_check}")
        return {"run_task_id": run_task_id, "initial_boundary": first,
                "warmup_control_steps": warmup_control_steps,
                "walking_boundaries": boundaries, "native_goal_diagnostic": native_check,
                "stabilization": stabilization,
                "finish_rejections": finish_rejections, "finish_result": finish_result,
                "recovery_lease": recovery_lease,
                "updates": events}
    finally:
        if initialized:
            await session.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--twist", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--expected-stop", choices=("forward_proximity", "command_segment_complete",
                                                 "motion_stalled", "external_contact"),
                        default="forward_proximity")
    parser.add_argument("--segment-limit", type=int, default=6)
    parser.add_argument("--warmup-control-steps", type=int, default=75)
    parser.add_argument("--stabilize-control-steps", type=int, default=0)
    parser.add_argument("--require-goal", action="store_true")
    parser.add_argument("--disconnect-on-guard", action="store_true")
    parser.add_argument("--test-recovery-lease", action="store_true")
    parser.add_argument("--test-finish-boundary", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Acceptance output must be new")
    root = Path(__file__).resolve().parents[1]
    if args.warmup_control_steps < 75:
        raise ValueError("Warmup requires the initial bounded velstand segment")
    if args.stabilize_control_steps and not 5 <= args.stabilize_control_steps <= 100:
        raise ValueError("Physical stabilization requires 5 to 100 controls")
    if args.test_finish_boundary and args.stabilize_control_steps < 5:
        raise ValueError("Finish boundary acceptance requires executed zero-command controls")
    result = asyncio.run(run(root, tuple(args.twist), args.expected_stop,
                             args.segment_limit, args.warmup_control_steps,
                             args.stabilize_control_steps, args.require_goal,
                             args.disconnect_on_guard, args.test_recovery_lease,
                             args.test_finish_boundary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.disconnect_on_guard:
        print(json.dumps({"run_task_id": result["run_task_id"],
                          "disconnect_boundary": result["disconnect_boundary"],
                          "control_steps": result["control_steps"]}, allow_nan=False))
    else:
        print(json.dumps({"run_task_id": result["run_task_id"],
                          "initial_control_steps": result["initial_boundary"]["control_steps"],
                          "walking_boundaries": result["walking_boundaries"],
                          "native_goal_diagnostic": result["native_goal_diagnostic"],
                          "stabilization": result["stabilization"],
                          "finish_rejections": result["finish_rejections"],
                          "finish_result": result["finish_result"],
                          "recovery_lease": result["recovery_lease"]}, allow_nan=False))


if __name__ == "__main__":
    main()
