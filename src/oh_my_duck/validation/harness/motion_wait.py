import argparse
import asyncio
import base64
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from uuid import uuid4

from oh_my_duck.core.paths import project_root
from oh_my_duck.validation.harness.control import available_port, control, expect_control_error


def save(path, value):
    with path.open("x") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")


async def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU motion wait validation requires empty CUDA_VISIBLE_DEVICES")
    import torch

    from oh_my_duck.integrations.edh.session import MicroDuckWorkerSession
    from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION

    root = project_root()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    configuration = {"backend": "cpu-mujoco-bam", "native_task_id": "motion-wait-tools",
        "catalog_dir": str(args.catalog.resolve(strict=True)), "seed": 20260929,
        "goal": {"kind": "object", "object": "ball_0", "distance_m": 0.5, "hold_ticks": 5},
        "spawn_pose": {"x_m": -1.6, "y_m": -0.5, "yaw_rad": math.pi},
        "task_instruction": "使用当前观测接近目标并检查运动等待期间的实际进度",
        "policy_revision": OFFICIAL_REVISION, "control_port": port, "control_secret": secret}
    events, samples, tools, frames = [], [], [], []
    phase = "warmup"
    record = {"passed": False, "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_has_tracked_changes": bool(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": {key: value for key, value in configuration.items() if key != "control_secret"},
        "gpu_acceptance_performed": False, "model_task_acceptance_performed": False}

    async def emit(message):
        events.append(message)
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            samples.append({**session._environment._navigation_samples[observation.observation_id],
                            "phase": phase, "control": data.get("control"),
                            "body_twist": list(observation.state["body_twist"])})
        elif message["event"] == "frame":
            name = f"frames/{len(frames):06d}.png"
            (args.output / name).write_bytes(base64.b64decode(
                data["observation"]["images"]["observer_follow"], validate=True))
            frames.append({"file": name, "simulation_time_s": data["simulation_time_s"], "phase": phase})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, arguments):
        started = time.monotonic()
        result = await control(port, secret, task_id, operation, arguments)
        if operation in {"progress", "wait_for_motion"} and result["metric_motion"] is not None:
            motion = result["metric_motion"]
            assert motion["sequence"] == result["sequence"]
            assert motion["body_position_m"] == result["body_position_m"]
            assert motion["stopped_samples"] == result["stopped_samples"]
        tools.append({"operation": operation, "arguments": arguments, "result": deepcopy(result),
                      "phase": phase, "elapsed_s": time.monotonic() - started})
        print(json.dumps({"operation": operation, "phase": phase, "sequence": result.get("sequence"),
                          "motion_wait": result.get("motion_wait")}), flush=True)
        return result

    async def wait(arguments):
        result = await call("wait_for_motion", arguments)
        while not result["motion_wait"]["boundary_confirmed"]:
            assert result["execution"]["state"] in {"running", "pausing"}
            result = await call("wait_for_motion", arguments)
        return result

    async def resume():
        status = session._status
        await session.resume({"owner_id": "planner", "execution_id": status["execution_id"],
            "boundary_id": status["boundary_event_id"], "state_version": status["state_version"]})

    async def unchanged_boundary(arguments):
        before = await call("progress", {})
        after = await call("wait_for_motion", arguments)
        assert after["motion_wait"]["boundary_confirmed"]
        for key in ("episode_id", "sequence", "stopped_samples", "goal_check", "body_position_m"):
            assert before[key] == after[key]
        for key in ("execution_id", "generation", "boundary_id", "state", "device_confirmed", "remaining_actions"):
            assert before["execution"][key] == after["execution"][key]
        assert after["execution"]["remaining_wall_time_s"] <= before["execution"]["remaining_wall_time_s"]

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
            "native_task_id": configuration["native_task_id"], "policy_id": "official-microduck-onnx",
            "execution_mode": "policy", "schema_path": str(args.edh_source.resolve(strict=True) /
            "harness/contracts/schema/physical.schema.json"), "monitor_every_actions": 1,
            "policy_max_actions_per_inference": 1})
        from OpenGL import GL

        record["renderer"] = await session._device.on_owner(lambda: GL.glGetString(GL.GL_RENDERER).decode())
        assert "llvmpipe" in record["renderer"].lower()
        await session.open_task({"native_task_id": configuration["native_task_id"], "run_task_id": task_id,
                                 "catalog_task_id": configuration["native_task_id"]})
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
            "goal_id": "motion-wait-tools", "attempt_id": "motion-wait", "instruction": configuration["task_instruction"],
            "entities": {"robot": "microduck"}, "required_capabilities": ["policy-navigation"],
            "success_contract": {"id": "motion-wait-tools", "version": "1",
                "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                "source": {"kind": "benchmark", "reference": "native-motion-wait-tools"}},
            "budget": {"max_control_steps": 1000, "max_wall_time_s": 600}, "context_refs": [],
            "decision_owner_id": "planner", "owner_assignment_id": "motion-wait-tools", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": configuration["native_task_id"]})
        first = await call("wait_for_motion", {"max_wait_s": 1})
        assert first["execution"]["state"] == "running" and not first["motion_wait"]["boundary_confirmed"]
        pump = session._pump
        assert pump is not None and not pump.done()
        cancelled_wait = asyncio.create_task(session._wait_motion_boundary(90))
        await asyncio.sleep(0)
        cancelled_wait.cancel()
        cancellation, = await asyncio.gather(cancelled_wait, return_exceptions=True)
        assert isinstance(cancellation, asyncio.CancelledError) and not pump.done()
        second = await call("wait_for_motion", {"max_wait_s": 1})
        assert second["sequence"] > first["sequence"] and session._pump is pump and not pump.cancelled()
        initial = await wait({})
        assert initial["sequence"] == 75 and initial["stopped_samples"] >= 5 and not initial["fallen"]
        record["cancelled_wait_preserved_motion"] = True
        await unchanged_boundary({"max_wait_s": 1})
        for arguments in ({"max_wait_s": True}, {"max_wait_s": False}, {"max_wait_s": 0},
                          {"max_wait_s": 91}, {"max_wait_s": 1.0}, {"max_wait_s": "1"}, {"extra": 1}):
            await expect_control_error(port, secret, task_id, "wait_for_motion", arguments,
                                       "未知字段" if "extra" in arguments else "max_wait_s")
        after_invalid = await call("progress", {})
        assert after_invalid["sequence"] == initial["sequence"] and after_invalid["goal_check"] == initial["goal_check"]
        record["invalid_requests_rejected"] = 7
        phase = "approach"
        await call("inspect_scene", {"prompt": "ball", "source": "simulator_ground_truth"})
        await call("approach_object", {"target_id": "ball_0", "standoff_m": 0.4})
        await resume()
        pending = await call("wait_for_motion", {"max_wait_s": 1})
        assert not pending["motion_wait"]["boundary_confirmed"] and not pending["metric_motion"]["completed"]
        final = await wait({"max_wait_s": 90})
        motion = final["metric_motion"]
        assert motion["completed"] and motion["error"] <= motion["tolerance"]
        assert final["stopped_samples"] >= 5 and not final["fallen"]
        assert final["goal_check"]["complete"]
        await unchanged_boundary({})
        phase = "hold"
        await call("set_command", {"command": {"twist": [0, 0, 0]}, "max_control_steps": 50})
        await resume()
        final = await wait({})
        assert final["stopped_samples"] >= 50 and final["goal_check"]["complete"]
        await call("finish_policy", {key: final["execution"][key]
                                    for key in ("execution_id", "generation", "boundary_id")})
        phase = "ended"
        await unchanged_boundary({"max_wait_s": 90})
        record.update(control_steps=final["sequence"], raw_sim_steps=session._status["raw_sim_steps"],
            final_progress=final, approach=motion, terminal_boundary=session._gate.snapshot(),
            cuda_initialized=torch.cuda.is_initialized())
        assert not record["cuda_initialized"] and record["raw_sim_steps"] == record["control_steps"] * 4
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        closed = await session.close()
        record["resources_released"] = (closed["closed"]
            and (session._device is None or session._device._closed)
            and (session._control_server is None or not session._control_server.is_serving())
            and (session._policy_server is None or not session._policy_server.is_serving()))
        record["passed"] = error is None and record["resources_released"]
        for name, value in (("result", record), ("events", events), ("samples", samples), ("tools", tools), ("frames", frames)):
            save(args.output / (name + ".json"), value)
        assert record["resources_released"]
    return record


def main():
    parser = argparse.ArgumentParser(description="使用官方 CPU policy 检查等待窗口、取消调用与实际运动边界")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), allow_nan=False))


if __name__ == "__main__":
    main()
