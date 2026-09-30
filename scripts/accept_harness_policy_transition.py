from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import secrets
from uuid import uuid4

from accept_harness_motion_guard import available_port, control, expect_control_error
from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


async def wait_execution(session):
    async with asyncio.timeout(300):
        await session._pump
        await session._await_motion_cleanup()
    status = session._gate.snapshot()
    if status["state"] not in ("paused", "ended") or not status["device_confirmed"]:
        raise AssertionError("原生执行没有确认动作边界")
    return {**status, "control_steps": session._device.executed_actions,
            "raw_sim_steps": session._device.raw_sim_steps}


async def run(arguments, output):
    root = Path(__file__).resolve().parents[1]
    edh = arguments.edh_source.resolve(strict=True)
    configuration = {
        "backend": "cpu-mujoco-bam", "native_task_id": "official-apartment-office",
        "catalog_dir": str(arguments.policy_dir.resolve(strict=True)),
        "seed": 20260929, "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
        "task_instruction": "执行指定策略动作，接续站立并检查实际停止",
        "policy_revision": OFFICIAL_REVISION,
    }
    if arguments.scene_config is not None:
        configuration.update(json.loads(arguments.scene_config.resolve(strict=True).read_text()))
        configuration["native_task_id"] = configuration["scene_id"]
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    configuration.update(control_port=port, control_secret=secret)
    updates = []

    async def emit(message):
        if message["event"] == "update":
            status = message["data"]["status"]
            updates.append({"state": status["state"], "control_steps": status["control_steps"],
                            "raw_sim_steps": status["raw_sim_steps"]})

    session = MicroDuckWorkerSession(emit)
    initialized = False
    evidence = {"status": "running", "policy": arguments.policy, "backend": configuration["backend"],
                "scene": configuration["native_task_id"], "policy_revision": OFFICIAL_REVISION}

    def record():
        output.write_text(json.dumps({**evidence, "updates": updates}, indent=2, allow_nan=False) + "\n")

    async def start(attempt):
        request = {
            "schema_version": "physical.subgoal.v1", "task_id": task_id,
            "team_run_id": task_id, "goal_id": "policy-transition", "attempt_id": attempt,
            "instruction": configuration["task_instruction"],
            "entities": {"robot": "microduck"}, "required_capabilities": ["policy-navigation"],
            "success_contract": {"id": "microduck-native-goal", "version": "1",
                                 "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                 "source": {"kind": "benchmark", "reference": "native-policy-transition"}},
            "budget": {"max_control_steps": 1000, "max_wall_time_s": 600},
            "context_refs": [], "decision_owner_id": "planner",
            "owner_assignment_id": "policy-transition-acceptance", "idempotency_key": uuid4().hex,
        }
        await session.start({"request": request, "native_task_id": configuration["native_task_id"],
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})

    async def resume():
        prior = session._status
        await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                              "boundary_id": prior["boundary_event_id"],
                              "state_version": prior["state_version"]})
        return await wait_execution(session)

    try:
        await session.initialize({
            "provider": "microduck", "scene_configuration": configuration,
            "native_task_id": configuration["native_task_id"], "policy_id": "official-microduck-onnx",
            "execution_mode": "policy", "schema_path": str(edh / "harness/contracts/schema/physical.schema.json"),
            "monitor_every_actions": 1, "policy_max_actions_per_inference": 1,
        })
        initialized = True
        await session.open_task({"native_task_id": configuration["native_task_id"],
                                 "run_task_id": task_id, "catalog_task_id": "policy-transition"})
        await start("skill")
        evidence["initial_boundary"] = await wait_execution(session)
        await expect_control_error(port, secret, task_id, "transition_policy",
                                   {"policy_name": "alpha_stand"}, "native episode termination")
        selected = await control(port, secret, task_id, "select_policy", {"policy_name": arguments.policy})
        evidence["selected"] = selected
        if selected["duration_s"] is None:
            raise ValueError("验收需要具有 manifest duration 的策略")
        required = round(selected["duration_s"] * 50)
        remaining = required
        while remaining:
            count = min(100, remaining)
            await control(port, secret, task_id, "set_command",
                          {"command": {"twist": [0, 0, 0]}, "max_control_steps": count})
            boundary = await resume()
            remaining -= count
            evidence["skill_boundary"] = boundary
            evidence["skill_progress"] = await control(port, secret, task_id, "progress", {})
            record()
            if boundary["control_steps"] != 75 + required - remaining:
                raise AssertionError("策略执行步数与原生 Gate 记录不符")
        if boundary["state"] != "ended" or boundary["stop_reason"] != "episode_terminated":
            raise AssertionError("完整策略没有产生原生终止边界")
        before = evidence["skill_progress"]
        transition = await control(port, secret, task_id, "transition_policy", {"policy_name": "alpha_stand"})
        after = await control(port, secret, task_id, "progress", {})
        evidence["transition"] = transition
        for key in ("episode_id", "sequence", "simulation_time_s", "body_position_m", "body_twist"):
            if before[key] != after[key]:
                raise AssertionError(f"策略转换改变了实际物理状态：{key}")
        if transition["transition"]["physical_stop_confirmed"]:
            raise AssertionError("策略转换错误地确认了物理停止")
        await control(port, secret, task_id, "set_command",
                      {"command": {"twist": [0, 0, 0]}, "max_control_steps": 100})
        await start("recovery")
        boundary = await wait_execution(session)
        evidence["recovery_boundary"] = boundary
        evidence["recovery_progress"] = await control(port, secret, task_id, "progress", {})
        record()
        if boundary["state"] != "paused" or boundary["control_steps"] != 100:
            raise AssertionError("接续站立没有执行完整的实际控制步")
        finish = await control(port, secret, task_id, "finish_policy",
                               {key: boundary[key] for key in ("execution_id", "generation", "boundary_id")})
        evidence["finish"] = finish
        evidence["status"] = "passed"
        record()
        return {"status": "passed", "policy": arguments.policy,
                "skill_control_steps": required, "recovery_control_steps": 100,
                "stopped_samples": finish["stop_confirmation"]["stopped_samples"]}
    finally:
        if initialized:
            await session.close()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=("ground_pick", "roulade", "kick_left", "kick_right"))
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--policy-dir", type=Path, default=root / ".cache/official-policies" / OFFICIAL_REVISION)
    parser.add_argument("--edh-source", type=Path, default=root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(run(arguments, arguments.output)), allow_nan=False))


if __name__ == "__main__":
    main()
