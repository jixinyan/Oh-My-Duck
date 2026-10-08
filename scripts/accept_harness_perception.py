import argparse
import asyncio
import json
from pathlib import Path
import secrets
import subprocess
from uuid import uuid4

import numpy as np
import torch

from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
from oh_my_duck.validation.harness.control import available_port, control, expect_control_error, wait_boundary


async def run_case(arguments, configured):
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    configuration = {
        "backend": "cpu-mujoco-bam", "native_task_id": "official-apartment-office",
        "catalog_dir": str(arguments.policy_dir.resolve(strict=True)),
        "seed": 20260929, "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
        "task_instruction": "读取当前公寓中的目标和感知能力",
        "policy_revision": OFFICIAL_REVISION, "control_port": port, "control_secret": secret,
    }
    if configured:
        configuration["perception_endpoint"] = f"http://127.0.0.1:{available_port()}"
    events = []

    async def emit(message):
        events.append(message)

    session = MicroDuckWorkerSession(emit)
    expected = ["simulator_ground_truth", "models"] if configured else ["simulator_ground_truth"]
    try:
        initialized = await session.initialize({
            "provider": "microduck", "scene_configuration": configuration,
            "native_task_id": configuration["native_task_id"], "policy_id": "official-microduck-onnx",
            "execution_mode": "policy",
            "schema_path": str(arguments.edh_source.resolve(strict=True) / "harness/contracts/schema/physical.schema.json"),
        })
        assert initialized["scene_metadata"]["perception_sources"] == expected
        await session.open_task({"native_task_id": configuration["native_task_id"],
                                 "run_task_id": task_id, "catalog_task_id": "perception-acceptance"})
        scene = await control(port, secret, task_id, "scene", {})
        assert scene["perception_sources"] == expected
        await session.start({"native_task_id": configuration["native_task_id"], "request": {
            "schema_version": "physical.subgoal.v1", "task_id": task_id,
            "team_run_id": task_id, "goal_id": "office-reached", "attempt_id": "perception",
            "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
            "required_capabilities": ["head-rgb"],
            "success_contract": {"id": "microduck-native", "version": "1",
                                 "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                 "source": {"kind": "benchmark", "reference": "perception-acceptance"}},
            "budget": {"max_control_steps": 200, "max_wall_time_s": 300}, "context_refs": [],
            "decision_owner_id": "planner", "owner_assignment_id": "perception-acceptance",
            "idempotency_key": uuid4().hex,
        }})
        boundary = await wait_boundary(session)
        assert boundary["control_steps"] == 75 and boundary["raw_sim_steps"] == 300
        backend = session._environment._backend()

        def physical_state():
            return (backend.data.qpos.copy(), backend.data.qvel.copy(), backend.data.ctrl.copy(),
                    backend._sequence, float(backend.data.time), backend._stopped_samples)

        before = await session._device.on_owner(physical_state)
        rejected = []
        for operation in ("inspect_scene", "observe"):
            for source in ("unavailable", "models") if not configured else ("unavailable",):
                await expect_control_error(port, secret, task_id, operation,
                                           {"prompt": "objects", "source": source}, "available sources:")
                rejected.append({"operation": operation, "source": source})
            if configured:
                # 实际连接关闭的本地端口，确认模型服务错误通过原生工具传递。
                await expect_control_error(port, secret, task_id, operation,
                                           {"prompt": "objects", "source": "models"}, "Connection refused")
        inspection = await control(port, secret, task_id, "inspect_scene",
                                   {"prompt": "objects", "source": "simulator_ground_truth"})
        observed = await control(port, secret, task_id, "observe",
                                 {"prompt": "objects", "source": "simulator_ground_truth"})
        assert inspection["targets"] and observed["perception"]["targets"]
        assert inspection["sequence"] == observed["sequence"] == 75
        assert all(target["detection_source"] == "simulator_ground_truth" for target in inspection["targets"])
        after = await session._device.on_owner(physical_state)
        assert all(np.array_equal(a, b) for a, b in zip(before, after))
        assert not torch.cuda.is_initialized()
        result = {"configured": configured, "perception_sources": expected,
                  "control_steps": 75, "raw_sim_steps": 300, "rejected": rejected,
                  "actual_model_connection_errors": 2 if configured else 0,
                  "targets": inspection["targets"], "physical_state_unchanged": True,
                  "native_events": len(events)}
    finally:
        await session.close()
    assert session._control_server.is_serving() is False
    assert session._policy_server.is_serving() is False
    result["resources_closed"] = True
    return result


async def run(arguments):
    results = [await run_case(arguments, configured) for configured in (False, True)]
    root = Path(__file__).resolve().parents[1]
    result = {"passed": True, "cases": results, "gpu_acceptance_performed": False,
              "cuda_initialized": torch.cuda.is_initialized(),
              "formal_navigation_acceptance_performed": False,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    with arguments.output.open("x") as record:
        json.dump(result, record, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "cases": len(results), "control_steps": 150,
                      "raw_sim_steps": 600, "resources_closed": True, "cuda_initialized": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
