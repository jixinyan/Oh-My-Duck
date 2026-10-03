import argparse
import asyncio
import base64
import json
from pathlib import Path
import secrets
from uuid import uuid4

from accept_harness_motion_guard import available_port, control, wait_boundary
from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


async def run(args):
    root = Path(__file__).resolve().parents[1]
    configuration = json.loads(args.scene_config.read_text())
    for name in ("usd_path", "provenance_path", "public_map_path"):
        configuration[name] = str((root / configuration[name]).resolve(strict=True))
    port, secret, task_id = available_port(), secrets.token_hex(32), "metric-" + uuid4().hex
    configuration.update(native_task_id="metric-tools", catalog_dir=str(args.catalog.resolve()),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret,
                         seed=20260930)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    events, samples, tools, frames = [], [], [], []
    phase = "warmup"

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
        print(json.dumps(tools[-1]), flush=True)
        return result

    async def resume():
        prior = session._status
        await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                              "boundary_id": prior["boundary_event_id"], "state_version": prior["state_version"]})
        await wait_boundary(session)
        for _ in range(3):
            progress = await call("progress", {})
            if progress["stopped_samples"] >= 5:
                break
            await call("set_command", {"command": {"twist": [0, 0, 0]}, "max_control_steps": 75})
            await resume()
        else:
            raise AssertionError("Warmup did not establish five actual stopped samples")

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
                                        "source": {"kind": "benchmark", "reference": "office-native-gt"}},
                   "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
                   "owner_assignment_id": "metric-acceptance", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "metric-tools",
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
        await wait_boundary(session)
        for operation, parameters in (("walk", {"distance_m": args.distance}), ("rotate", {"angle_deg": args.angle})):
            phase = operation
            await call(operation, parameters)
            await resume()
            progress = await call("progress", {})
            evidence = progress["metric_motion"]
            if evidence is None or not evidence["completed"] or evidence["error"] > evidence["tolerance"]:
                raise AssertionError(f"Metric target failed: {evidence}; guard={progress['motion_guard']}")
            if progress["stopped_samples"] < 5 or progress["fallen"]:
                raise AssertionError("Metric target lacks measured upright stop")
        final = await call("progress", {})
        await call("finish_policy", {key: final["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
        sensor = await call("read_sensor", {"sensor": "head_rgb"})
        (args.output / "head-final.png").write_bytes(base64.b64decode(sensor["measurements"]["rgb_png_base64"], validate=True))
        (args.output / "result.json").write_text(json.dumps({"passed": True, "task_id": task_id, "tools": tools,
                                                             "frames": frames, "scope": "Real native ActionGate and Newton policy tools; model acceptance separate"}, indent=2) + "\n")
    finally:
        (args.output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
        (args.output / "events.json").write_text(json.dumps(events) + "\n")
        (args.output / "tools.json").write_text(json.dumps(tools, indent=2) + "\n")
        (args.output / "frames.json").write_text(json.dumps(frames, indent=2) + "\n")
        await session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.4)
    parser.add_argument("--angle", type=float, default=45)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
