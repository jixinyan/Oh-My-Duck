import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import secrets
import subprocess
from uuid import uuid4

from oh_my_duck.core.paths import project_root
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


async def native_continuation(args, output):
    from oh_my_duck.integrations.edh.session import MicroDuckWorkerSession
    from oh_my_duck.validation.harness.control import available_port, control, wait_boundary

    port, secret = available_port(), secrets.token_hex(32)
    configuration = {
        "backend": "cpu-mujoco-bam", "native_task_id": "native-continuation",
        "catalog_dir": str(args.catalog.resolve(strict=True)), "seed": 20261009,
        "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        "task_instruction": "检查原生任务绑定、实际控制、独立目标记录与连续物理状态",
        "control_port": port, "control_secret": secret,
    }
    statuses = []

    async def emit(message):
        if message["event"] == "update":
            statuses.append(message["data"]["status"])

    session = MicroDuckWorkerSession(emit)
    records, previous = [], None
    released = False
    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
                                  "native_task_id": "native-continuation", "policy_id": "official-microduck-onnx",
                                  "execution_mode": "policy",
                                  "schema_path": str(args.edh_source / "harness/contracts/schema/physical.schema.json"),
                                  "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        for _ in range(2):
            task_id = uuid4().hex
            await session.open_task({"native_task_id": "native-continuation", "run_task_id": task_id,
                                     "catalog_task_id": "native-continuation"})
            initial = await control(port, secret, task_id, "progress", {})
            start = initial["task_start"]
            assert start["run_task_id"] == task_id and initial["execution"] is None
            assert start["sequence"] == initial["sequence"] and start["episode_id"] == initial["episode_id"]
            assert start["goal_check"]["checks"]["goal_reached"]["evidence"]["held_ticks"] == 0
            if previous is not None:
                assert start["sequence"] == previous["sequence"] and start["episode_id"] == previous["episode_id"]
                assert start["body_position_m"] == previous["body_position_m"]
                assert start["body_twist"] == previous["body_twist"]
                assert start["goal_check"]["checks"]["goal_reached"]["evidence"]["goal_scope_id"] != (
                    previous["goal_check"]["checks"]["goal_reached"]["evidence"]["goal_scope_id"])
            request = {
                "schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
                "goal_id": "native-continuation", "attempt_id": "attempt-1",
                "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
                "required_capabilities": ["policy-navigation"],
                "success_contract": {"id": "native-continuation", "version": "1",
                                     "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                     "source": {"kind": "benchmark", "reference": "native-continuation"}},
                "budget": {"max_control_steps": 100, "max_wall_time_s": 300},
                "context_refs": [], "decision_owner_id": "planner",
                "owner_assignment_id": "continuation-acceptance", "idempotency_key": uuid4().hex,
            }
            await session.start({"request": request, "native_task_id": "native-continuation",
                                 "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
            await wait_boundary(session)
            progress = await control(port, secret, task_id, "progress", {})
            assert progress["task_start"] == start and progress["sequence"] - start["sequence"] == 75
            assert session._device.executed_actions == 75 and session._device.raw_sim_steps == 300
            assert progress["stopped_samples"] >= 5
            finish = await control(port, secret, task_id, "finish_policy", {
                key: progress["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
            assert finish["execution"]["state"] == "ended" and finish["execution"]["device_confirmed"]
            assert finish["execution"]["stop_reason"] == "policy_stop"
            previous = await control(port, secret, task_id, "progress", {})
            records.append({"initial": initial, "final": previous, "finish": finish})
            await session.close_task()
    finally:
        closed = await session.close()
        released = (closed["closed"] and (session._device is None or session._device._closed) and
                    (session._control_server is None or not session._control_server.is_serving()) and
                    (session._policy_server is None or not session._policy_server.is_serving()))
        (output / "native-tasks.json").write_text(json.dumps(
            {"tasks": records, "execution_updates": statuses, "resources_released": released},
            ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if not released:
            raise AssertionError("原生任务检查未释放全部会话资源")
    return {"passed": True, "tasks": len(records), "policy_control_steps": 150,
            "raw_sim_steps": 600, "resources_released": released}


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU task continuation requires CUDA_VISIBLE_DEVICES to be empty")
    import numpy as np
    import torch

    from oh_my_duck.integrations.edh.environment import MicroDuckEnvironment
    from oh_my_duck.robotics.backends.simulation import MotionBusyError

    root = project_root()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "backend": "cpu-mujoco-bam", "native_task_id": "goal-continuation",
        "catalog_dir": str(args.catalog.resolve(strict=True)), "seed": 20261009,
        "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 2.0, "y_m": 0.0, "yaw_rad": 0.0},
        "task_instruction": "Measure independent goal evidence across retained tasks.",
    }
    difference = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    report = {"passed": False, "scope": "CPU retained goal evidence lifecycle",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_has_tracked_changes": bool(difference),
              "source_diff_sha256": hashlib.sha256(difference).hexdigest(),
              "policy_revision": OFFICIAL_REVISION,
              "configuration": configuration,
              "runtime_versions": {name: version(name) for name in
                                   ("mujoco", "better-actuator-models", "onnxruntime", "numpy")},
              "gpu_acceptance_performed": False, "agent_task_acceptance_performed": False}
    environment = MicroDuckEnvironment(configuration)

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def physical_state():
        backend = environment._backend()
        return {"episode_id": backend.episode_id, "sequence": backend._sequence,
                "simulation_time_s": float(backend.data.time), "policy": backend.active_policy.name,
                "qpos": backend.data.qpos.copy(), "qvel": backend.data.qvel.copy(),
                "ctrl": backend.data.ctrl.copy(), "last_action": backend.policy_inference.last_action.copy()}

    try:
        environment.reset("goal-continuation", configuration)
        backend = environment._backend()
        from OpenGL import GL

        report["renderer"] = GL.glGetString(GL.GL_RENDERER).decode("utf-8")
        if not any(name in report["renderer"].lower() for name in ("llvmpipe", "softpipe")):
            raise RuntimeError("CPU continuation requires a Mesa software renderer")
        original = backend.check_goal()
        try:
            environment.bind_task("another-task")
        except ValueError:
            assert backend.check_goal() == original
        else:
            raise AssertionError("Another task identity was admitted")
        backend.infer_policy()
        try:
            environment.bind_task("goal-continuation")
        except MotionBusyError:
            assert backend.check_goal() == original
        else:
            raise AssertionError("Task binding admitted an unsettled policy action")
        backend.discard_pending_inference()
        report["foreign_task_rejected"] = True
        report["pending_action_rejected"] = True
        scopes = []
        for index in range(2):
            before = physical_state()
            environment.bind_task("goal-continuation")
            after = physical_state()
            for key in before:
                if isinstance(before[key], np.ndarray):
                    np.testing.assert_array_equal(before[key], after[key])
                elif before[key] != after[key]:
                    raise AssertionError(f"Task binding changed physical state: {key}")
            initial = backend.check_goal()
            evidence = initial["checks"]["goal_reached"]["evidence"]
            assert not initial["complete"] and evidence["held_ticks"] == 0
            assert evidence["goal_bound_sequence"] == before["sequence"]
            assert backend._goal == configuration["goal"]
            for _ in range(10):
                environment.observe()
                assert not environment.check(["goal_reached"])[0].value
                assert backend.check_goal() == initial
            requests = []
            for step in range(75):
                ticket = backend.infer_policy()
                request_id = uuid4().hex
                applied = backend.apply_policy_action(ticket["action"], request_id=request_id,
                                                      expected_sequence=backend._sequence)
                assert applied["raw_sim_steps"] == 4 and not applied["interrupted"]
                if step < 4:
                    assert not backend.check_goal()["complete"]
                requests.append({"inference": ticket, "applied": applied})
            final = backend.check_goal()
            assert final["complete"] and environment.check(["goal_reached"])[0].value
            assert final["sequence"] - initial["sequence"] == 75
            assert final["checks"]["goal_reached"]["evidence"]["held_ticks"] >= 5
            assert backend._stopped_samples >= 5
            (output / f"actions-{index + 1}.json").write_text(json.dumps(requests, allow_nan=False) + "\n")
            scopes.append({"initial": initial, "final": final, "physical_state_preserved": True,
                           "repeated_reads_verified": 10, "control_steps": 75, "raw_sim_steps": 300})
            report["tasks"] = scopes
            save()
        assert scopes[0]["final"]["sequence"] == scopes[1]["initial"]["sequence"]
        assert scopes[0]["initial"]["checks"]["goal_reached"]["evidence"]["goal_scope_id"] != (
            scopes[1]["initial"]["checks"]["goal_reached"]["evidence"]["goal_scope_id"])
    finally:
        environment.close()
        report["resources_released"] = True
        report["cuda_initialized"] = torch.cuda.is_initialized()
        save()
    report["native_session"] = asyncio.run(native_continuation(args, output))
    report["passed"] = True
    save()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))
