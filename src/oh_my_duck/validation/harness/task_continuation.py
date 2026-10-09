import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import torch

from oh_my_duck.core.paths import project_root
from oh_my_duck.integrations.edh.environment import MicroDuckEnvironment
from oh_my_duck.robotics.backends.simulation import MotionBusyError
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU task continuation requires CUDA_VISIBLE_DEVICES to be empty")
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
        report["passed"] = True
    finally:
        environment.close()
        report["resources_released"] = True
        report["cuda_initialized"] = torch.cuda.is_initialized()
        save()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))
