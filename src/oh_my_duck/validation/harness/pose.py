import argparse
import asyncio
import base64
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

from oh_my_duck.core.paths import project_root


PHASES = (
    ("neutral", 0.0, 0.0),
    ("positive", 0.2, 0.01),
    ("return_positive", 0.0, 0.0),
    ("negative", -0.2, -0.01),
    ("return_negative", 0.0, 0.0),
)


async def run(args):
    from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
    from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
    from oh_my_duck.validation.harness.control import available_port, control, wait_boundary
    from oh_my_duck.validation.harness.pose_records import verify

    root = project_root()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    scene_path = args.scene_config.resolve(strict=True)
    configuration = json.loads(scene_path.read_text())
    if configuration["backend"] != "cpu-mujoco-bam":
        raise ValueError("This pose campaign requires the official CPU apartment")
    port, secret, task_id = available_port(), secrets.token_hex(32), "pose-" + uuid4().hex
    configuration.update(native_task_id="pose-tools", catalog_dir=str(args.catalog.resolve(strict=True)),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret,
                         seed=args.seed)
    difference = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    record = {"passed": False, "task_id": task_id, "scope": "Native CPU pose commands and physical response",
              "provenance": {
                  "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "source_has_tracked_changes": bool(difference),
                  "source_diff_sha256": hashlib.sha256(difference).hexdigest(),
                  "runner_source_path": "src/oh_my_duck/validation/harness/pose.py",
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "scene_config_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                  "catalog_manifest_sha256": hashlib.sha256((args.catalog / "manifest.json").read_bytes()).hexdigest(),
                  "policy_revision": OFFICIAL_REVISION,
                  "runtime_versions": {name: version(name) for name in
                                       ("mujoco", "better-actuator-models", "onnxruntime", "numpy")},
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "configuration": {key: value for key, value in configuration.items() if key != "control_secret"},
              }}
    (args.output / "provenance.json").write_text(json.dumps(record["provenance"], indent=2, allow_nan=False) + "\n")
    events, samples, tools, frames = [], [], [], []
    phase = "warmup"

    async def emit(message):
        events.append(message)
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            sample = session._environment._navigation_samples[observation.observation_id]
            samples.append({**sample, "phase": phase, "state": data["status"]["state"],
                            "control": data.get("control"),
                            "joint_position_rad": list(observation.state["joint_position_rad"]),
                            "joint_velocity_rad_s": list(observation.state["joint_velocity_rad_s"]),
                            "body_twist": list(observation.state["body_twist"])})
        elif message["event"] == "frame":
            relative = f"frames/{len(frames):06d}.png"
            image = data["observation"]["images"]["observer_follow"]
            (args.output / relative).write_bytes(base64.b64decode(image, validate=True))
            frames.append({"file": relative, "phase": phase, "simulation_time_s": data["simulation_time_s"]})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, arguments):
        result = await control(port, secret, task_id, operation, arguments)
        tools.append({"operation": operation, "arguments": arguments, "result": result, "phase": phase})
        return result

    def save():
        for name, value in (("result", record), ("samples", samples), ("events", events),
                            ("tools", tools), ("frames", frames)):
            (args.output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
                                  "native_task_id": "pose-tools", "policy_id": "official-microduck-onnx",
                                  "execution_mode": "policy",
                                  "schema_path": str(args.edh_source / "harness/contracts/schema/physical.schema.json"),
                                  "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        await session.open_task({"native_task_id": "pose-tools", "run_task_id": task_id, "catalog_task_id": "pose-tools"})
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
                   "goal_id": "pose-tools", "attempt_id": "attempt-1",
                   "instruction": "Measure official head pitch and body height commands through native control tools.",
                   "entities": {"robot": "microduck"}, "required_capabilities": ["policy-navigation"],
                   "success_contract": {"id": "pose-tools", "version": "1",
                                        "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                        "source": {"kind": "benchmark", "reference": configuration["scene_id"] + "-native-gt"}},
                   "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
                   "owner_assignment_id": "pose-acceptance", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "pose-tools",
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
        await wait_boundary(session)
        selected = await call("select_policy", {"policy_name": "alpha_stand"})
        record["policy_selection"] = selected
        for phase, head_pitch, body_height in PHASES:
            await call("set_command", {"command": {"twist": [0.0] * 3,
                "head": [0.0, head_pitch, 0.0, 0.0], "body": [0.0, 0.0, body_height, 0.0, 0.0, 0.0]},
                "max_control_steps": 100})
            prior = session._status
            await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                                  "boundary_id": prior["boundary_event_id"], "state_version": prior["state_version"]})
            await wait_boundary(session)
            progress = await call("progress", {})
            measured = await call("observe", {})
            await call("read_sensor", {"sensor": "joint_state"})
            await call("read_sensor", {"sensor": "head_rgb"})
            print(json.dumps({"phase": phase, "sequence": progress["sequence"],
                              "height_m": progress["height_m"],
                              "head_pitch_rad": measured["measurements"]["joint_position_rad"][6]}, allow_nan=False), flush=True)
            save()
        final = await call("progress", {})
        record["termination"] = await call("finish_policy", {
            key: final["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
        record["terminal_boundary"] = session._gate.snapshot()
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        closed = await session.close()
        released = (closed["closed"] and (session._device is None or session._device._closed)
                    and (session._control_server is None or not session._control_server.is_serving())
                    and (session._policy_server is None or not session._policy_server.is_serving()))
        record.update(closed=closed, resources_released=released)
        import torch
        record["cuda_initialized"] = torch.cuda.is_initialized()
        save()
        if not released:
            raise AssertionError("Pose campaign did not release native session resources")
    audit = verify(args.output, args.catalog)
    record.update(passed=True, audit=audit)
    save()
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    args.edh_source = args.edh_source.resolve(strict=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU pose validation requires CUDA_VISIBLE_DEVICES to be empty")
    print(json.dumps(asyncio.run(run(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
