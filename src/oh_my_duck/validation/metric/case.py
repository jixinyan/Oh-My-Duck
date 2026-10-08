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
from oh_my_duck.validation.harness.control import available_port, control, wait_boundary
from oh_my_duck.validation.metric.plans import case_motions
from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def provenance(root, args, configuration):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    difference = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    return {"provenance_schema_version": 2,
            "runner_source_path": "src/oh_my_duck/validation/metric/case.py",
            "runner_entrypoint_sha256": hashlib.sha256((root / "scripts/accept_metric_policy_tools.py").read_bytes()).hexdigest(),
            "source_revision": revision, "source_diff_sha256": hashlib.sha256(difference).hexdigest(),
            "source_has_tracked_changes": bool(difference),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scene_config_sha256": hashlib.sha256(args.scene_config.read_bytes()).hexdigest(),
            "policy_revision": OFFICIAL_REVISION,
            "catalog_manifest_sha256": hashlib.sha256((args.catalog / "manifest.json").read_bytes()).hexdigest(),
            "runtime_versions": {name: version(name) for name in
                                 ("mujoco", "better-actuator-models", "onnxruntime", "numpy")},
            "python_version": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "onnxruntime_telemetry_disabled": os.environ.get("ORT_DISABLE_TELEMETRY") == "1",
            "configuration": {key: value for key, value in configuration.items()
                              if key != "control_secret"},
            "commands": {"motions": args.motions, "operations": args.operations, "distance_m": args.distance,
                         "angle_deg": args.angle, "speed_m_s": args.speed,
                         "angular_speed_deg_s": args.angular_speed}}


def require_upright_stop(progress):
    if progress["stopped_samples"] < 5 or progress["fallen"]:
        raise AssertionError("Metric target lacks measured upright stop")
    if progress["contact_evidence"]["non_ground_external_contact_samples_total"]:
        raise AssertionError("Metric motion contacted an external obstacle")


async def run(args):
    root = project_root()
    args.scene_config = args.scene_config if args.scene_config.is_absolute() else root / args.scene_config
    configuration = json.loads(args.scene_config.read_text())
    for name in ("usd_path", "provenance_path", "public_map_path"):
        if name in configuration:
            configuration[name] = str((root / configuration[name]).resolve(strict=True))
    port, secret, task_id = available_port(), secrets.token_hex(32), "metric-" + uuid4().hex
    configuration.update(native_task_id="metric-tools", catalog_dir=str(args.catalog.resolve()),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret,
                         seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    events, samples, tools, frames = [], [], [], []
    phase = "warmup"
    record = {"passed": False, "task_id": task_id,
              "scope": "Actual native ActionGate and measured policy tools; model task acceptance separate",
              "provenance": provenance(root, args, configuration)}
    (args.output / "provenance.json").write_text(json.dumps(record["provenance"], indent=2) + "\n")

    async def emit(message):
        events.append(message)
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            sample = session._environment._navigation_samples[observation.observation_id]
            samples.append({**sample, "phase": phase, "state": data["status"]["state"],
                            "control": data.get("control"), "body_twist": list(observation.state["body_twist"])})
        elif message["event"] == "frame":
            relative = f"frames/{len(frames):06d}.png"
            image = data["observation"]["images"]["observer_follow"]
            (args.output / relative).write_bytes(base64.b64decode(image, validate=True))
            frames.append({"file": relative, "phase": phase, "simulation_time_s": data["simulation_time_s"]})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, arguments):
        result = await control(port, secret, task_id, operation, arguments)
        tools.append({"operation": operation, "arguments": arguments, "result": result, "phase": phase})
        report = result if operation not in {"read_sensor", "observe"} else {
            "sensor": arguments.get("sensor", "combined"), "sequence": result["sequence"]}
        print(json.dumps({"operation": operation, "arguments": arguments,
                          "result": report, "phase": phase}), flush=True)
        return result

    async def resume():
        prior = session._status
        await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                              "boundary_id": prior["boundary_event_id"], "state_version": prior["state_version"]})
        first = await call("wait_for_motion", {})
        second = await call("wait_for_motion", {})
        if first["sequence"] != second["sequence"] or first["execution"]["boundary_id"] != second["execution"]["boundary_id"]:
            raise AssertionError("Waiting on an existing boundary changed the native execution")

    async def observe_boundary():
        first = await call("observe", {})
        second = await call("observe", {})
        for key in ("episode_id", "sequence", "body_position_m", "stopped_samples"):
            if first[key] != second[key]:
                raise AssertionError("Repeated boundary observation changed physical state or stopped samples")
        for key in ("joint_names", "joint_position_rad", "joint_velocity_rad_s", "odometry"):
            if first["measurements"][key] != second["measurements"][key]:
                raise AssertionError("Repeated boundary observation changed measured joint or odometry state")
        measurements = second["measurements"]
        if (len(set(measurements["joint_names"])) != 14 or len(measurements["joint_position_rad"]) != 14 or
                len(measurements["joint_velocity_rad_s"]) != 14 or len(measurements["tof_distance_mm"]) != 64):
            raise AssertionError("Combined observation lacks actual joint or ToF measurements")
        if second["execution"]["remaining_wall_time_s"] > first["execution"]["remaining_wall_time_s"]:
            raise AssertionError("Remaining native wall budget increased without a new execution")
        if not base64.b64decode(second["rgb_png_base64"], validate=True):
            raise AssertionError("Combined observation has no original head camera")
        return second

    try:
        edh = root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd"
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
                                  "native_task_id": "metric-tools", "policy_id": "official-microduck-onnx",
                                  "execution_mode": "policy", "schema_path": str(edh / "harness/contracts/schema/physical.schema.json"),
                                  "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        await session.open_task({"native_task_id": "metric-tools", "run_task_id": task_id, "catalog_task_id": "metric-tools"})
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
                   "goal_id": "metric-tools", "attempt_id": "attempt-1", "instruction": configuration["task_instruction"],
                   "entities": {"robot": "microduck"}, "required_capabilities": ["policy-navigation"],
                   "success_contract": {"id": "metric-tools", "version": "1",
                                        "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                        "source": {"kind": "benchmark", "reference": configuration["scene_id"] + "-native-gt"}},
                   "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
                   "owner_assignment_id": "metric-acceptance", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "metric-tools",
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
        await call("wait_for_motion", {})
        for _ in range(3):
            progress = await call("progress", {})
            if progress["fallen"]:
                raise AssertionError("Warmup encountered a fallen robot")
            if progress["stopped_samples"] >= 5:
                break
            await call("set_command", {"command": {"twist": [0, 0, 0]}, "max_control_steps": 75})
            await resume()
        progress = await call("progress", {})
        require_upright_stop(progress)
        await observe_boundary()
        measurements = []
        record["measurements"] = measurements
        for index, motion in enumerate(args.motions):
            operation, parameters = motion["operation"], motion["arguments"]
            phase = f"{index}-{operation}"
            await call(operation, parameters)
            await resume()
            progress = await call("progress", {})
            evidence = progress["metric_motion"]
            if evidence is None or not evidence["completed"] or evidence["error"] > evidence["tolerance"]:
                raise AssertionError(f"Metric target failed: {evidence}; guard={progress['motion_guard']}")
            require_upright_stop(progress)
            await observe_boundary()
            measurements.append({"operation": operation, "arguments": parameters,
                                 "evidence": evidence, "progress": progress})
        final = await call("progress", {})
        termination = await call("finish_policy", {key: final["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
        boundary = session._gate.snapshot()
        if boundary["state"] != "ended" or not boundary["device_confirmed"]:
            raise AssertionError("Metric acceptance has no confirmed terminal boundary")
        sensor = await call("read_sensor", {"sensor": "head_rgb"})
        (args.output / "head-final.png").write_bytes(base64.b64decode(sensor["measurements"]["rgb_png_base64"], validate=True))
        record.update(measurements=measurements, terminal_boundary=boundary,
                      termination=termination)
    finally:
        error = sys.exception()
        record["final_execution_status"] = session._status
        record["final_metric_motion"] = (session._metric_motion.result
                                          if session._metric_motion is not None else None)
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        record["passed"] = False
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        (args.output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
        (args.output / "events.json").write_text(json.dumps(events) + "\n")
        (args.output / "tools.json").write_text(json.dumps(tools, indent=2) + "\n")
        (args.output / "frames.json").write_text(json.dumps(frames, indent=2) + "\n")
        closed = await session.close()
        released = (closed["closed"] and
                    (session._device is None or session._device._closed) and
                    (session._control_server is None or not session._control_server.is_serving()) and
                    (session._policy_server is None or not session._policy_server.is_serving()))
        record.update(closed=closed, resources_released=released)
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        if not released:
            raise AssertionError("Metric acceptance did not release native session resources")
        record["passed"] = error is None
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.4)
    parser.add_argument("--angle", type=float, default=45)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--angular-speed", type=float, default=45)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--operations", nargs="+", choices=("walk", "rotate"), default=["walk", "rotate"])
    parser.add_argument("--sequence", type=Path)
    args = parser.parse_args()
    declared = ({"motions": json.loads(args.sequence.read_text())} if args.sequence is not None else {
        "operations": args.operations, "distance_m": args.distance, "angle_deg": args.angle,
        "speed_m_s": args.speed, "angular_speed_deg_s": args.angular_speed})
    args.motions = case_motions(declared)
    args.operations = [motion["operation"] for motion in args.motions]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
