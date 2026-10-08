import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

from oh_my_duck.core.paths import project_root


async def run(arguments):
    from OpenGL import GL
    import torch

    from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
    from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
    from oh_my_duck.validation.harness.control import available_port, control, expect_control_error, wait_boundary
    from oh_my_duck.validation.metric.case import require_upright_stop

    root = project_root()
    configuration = json.loads(arguments.scene_config.read_text())
    if configuration["backend"] != "cpu-mujoco-bam" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Metric admission validation requires CPU MuJoCo/BAM and empty CUDA_VISIBLE_DEVICES")
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    configuration.update(native_task_id="metric-admission", catalog_dir=str(arguments.catalog.resolve(strict=True)),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret,
                         seed=arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=False)
    events, samples, calls = [], [], []
    record = {"passed": False, "formal_navigation_acceptance_performed": False,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy_revision": OFFICIAL_REVISION, "seed": arguments.seed,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "scene_config_sha256": hashlib.sha256(arguments.scene_config.read_bytes()).hexdigest(),
              "catalog_manifest_sha256": hashlib.sha256((arguments.catalog / "manifest.json").read_bytes()).hexdigest(),
              "motions": [], "rejections": [], "explicit_replacements": []}

    async def emit(message):
        events.append(message)
        if message["event"] == "update" and message.get("data", {}).get("observation"):
            observation = session._latest_observation
            samples.append({**session._environment._navigation_samples[observation.observation_id],
                            "control": message["data"].get("control")})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, parameters):
        result = await control(port, secret, task_id, operation, parameters)
        calls.append({"operation": operation, "arguments": parameters, "result": result})
        return result

    async def resume():
        status = session._status
        await session.resume({"owner_id": "planner", "execution_id": status["execution_id"],
                              "boundary_id": status["boundary_event_id"], "state_version": status["state_version"]})
        await wait_boundary(session, timeout_s=None)

    async def unchanged_motion(prepared, motion):
        backend = session._environment._backend()

        def snapshot():
            state = backend._native_state()
            return {"sequence": backend._sequence, "stopped_samples": backend._stopped_samples,
                    "physical": {key: state[key] for key in ("body_position_m", "body_twist",
                        "joint_position_rad", "joint_velocity_rad_s", "odometry")},
                    "command": backend._requested_command,
                    "segment_request_id": session._motion_segment["request_id"],
                    "gate": {key: session._gate.snapshot()[key] for key in
                             ("state", "generation", "boundary_id", "device_confirmed")}}

        before = await session._device.on_owner(snapshot)
        for operation, parameters in (("walk", {"distance_m": 0.5}), ("rotate", {"angle_deg": 45})):
            await expect_control_error(port, secret, task_id, operation, parameters,
                                       f"Metric motion request {prepared['request_id']} is pending")
            after = await session._device.on_owner(snapshot)
            assert before == after and session._metric_motion is motion
            assert motion.request_id == prepared["request_id"] and motion.last_sequence == -1
            record["rejections"].append({"pending_operation": prepared["operation"],
                "pending_request_id": prepared["request_id"], "rejected_operation": operation,
                "physical_sequence": before["sequence"], "physical_state_unchanged": True,
                "command_and_native_boundary_unchanged": True})

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
            "native_task_id": "metric-admission", "policy_id": "official-microduck-onnx",
            "execution_mode": "policy", "monitor_every_actions": 1,
            "schema_path": str(arguments.edh_source.resolve(strict=True) / "harness/contracts/schema/physical.schema.json")})
        record["renderer"] = await session._device.on_owner(lambda: GL.glGetString(GL.GL_RENDERER).decode("utf-8"))
        if not any(name in record["renderer"].lower() for name in ("llvmpipe", "softpipe")):
            raise RuntimeError("Metric admission validation requires a Mesa software renderer")
        await session.open_task({"native_task_id": "metric-admission", "run_task_id": task_id,
                                 "catalog_task_id": "metric-admission"})
        await session.start({"native_task_id": "metric-admission", "request": {
            "schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
            "goal_id": "metric-admission", "attempt_id": "attempt-1",
            "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
            "required_capabilities": ["policy-navigation"], "success_contract": {
                "id": "metric-admission", "version": "1", "all": [
                    {"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                "source": {"kind": "benchmark", "reference": "metric-admission"}},
            "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
            "owner_assignment_id": "metric-admission", "idempotency_key": uuid4().hex}})
        await wait_boundary(session, timeout_s=None)
        require_upright_stop(await call("progress", {}))
        for operation, parameters in (("walk", {"distance_m": 0.5}), ("rotate", {"angle_deg": 45})):
            prepared = await call(operation, parameters)
            motion = session._metric_motion
            await unchanged_motion(prepared, motion)
            await resume()
            progress = await call("progress", {})
            evidence = progress["metric_motion"]
            assert evidence["request_id"] == prepared["request_id"]
            assert evidence["completed"] and evidence["error"] <= evidence["tolerance"]
            require_upright_stop(progress)
            record["motions"].append(evidence)
        for operation, parameters in (("walk", {"distance_m": 0.5}), ("rotate", {"angle_deg": -45})):
            prepared = await call(operation, parameters)
            await unchanged_motion(prepared, session._metric_motion)
            replacement = await call("set_command", {"command": {"twist": [0, 0, 0]}, "max_control_steps": 50})
            assert session._metric_motion is None
            assert replacement["effective_after_sequence"] == prepared["command_admission"]["effective_after_sequence"]
            await resume()
            progress = await call("progress", {})
            require_upright_stop(progress)
            assert progress["metric_motion"] is None
            record["explicit_replacements"].append({"prepared_request_id": prepared["request_id"],
                "operation": operation, "replacement_request_id": replacement["request_id"],
                "zero_control_steps": 50, "progress": progress})
        progress = await call("progress", {})
        record["termination"] = await call("finish_policy", {
            key: progress["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
        record["terminal_boundary"] = session._gate.snapshot()
        assert record["terminal_boundary"]["state"] == "ended"
        assert record["terminal_boundary"]["device_confirmed"]
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        record["final_execution_status"] = session._status
        for name, payload in (("events.json", events), ("samples.json", samples), ("tools.json", calls)):
            (arguments.output / name).write_text(json.dumps(payload, allow_nan=False) + "\n")
        (arguments.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        closed = await session.close()
        released = (closed["closed"] and (session._device is None or session._device._closed)
                    and (session._control_server is None or not session._control_server.is_serving())
                    and (session._policy_server is None or not session._policy_server.is_serving()))
        record.update(closed=closed, resources_released=released, passed=error is None and released)
        record["cuda_initialized"] = torch.cuda.is_initialized()
        assert not record["cuda_initialized"]
        (arguments.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        assert released
    print(json.dumps({"passed": True, "rejections": len(record["rejections"]),
                      "completed_motions": len(record["motions"]), "explicit_replacements": 2,
                      "control_steps": record["final_execution_status"]["control_steps"],
                      "resources_released": record["resources_released"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    asyncio.run(run(parser.parse_args()))
