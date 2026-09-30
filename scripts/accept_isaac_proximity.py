import argparse
import asyncio
import base64
import hashlib
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
    port, secret = available_port(), secrets.token_hex(32)
    task_id = "isaac-proximity-" + uuid4().hex
    configuration.update(native_task_id="office-proximity", catalog_dir=str(args.catalog.resolve()),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret)
    samples, frames, tools, boundaries = [], [], [], []
    phase = "warmup"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()

    async def emit(message):
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            if observation.observation_id != data["observation"]["observation_id"]:
                raise RuntimeError("Recorded native observation identity differs")
            navigation = session._environment._navigation_samples[observation.observation_id]
            if samples and navigation["sequence"] == samples[-1]["sequence"]:
                return
            samples.append({**navigation, "body_twist": observation.state["body_twist"],
                            "imu": observation.state["imu"], "phase": phase,
                            "gate_state": data["status"]["state"],
                            "simulation_time_s": navigation["sequence"] / 50,
                            "motion_guard": data.get("control", {}).get("motion_guard")})
        elif message["event"] == "frame":
            relative = f"frames/{len(frames):06d}.png"
            image = data["observation"]["images"]["observer_follow"]
            (args.output / relative).write_bytes(base64.b64decode(image, validate=True))
            frames.append({"file": relative, "simulation_time_s": data["simulation_time_s"],
                           "phase": phase, "observation_id": data["observation"]["observation_id"]})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, arguments):
        result = await control(port, secret, task_id, operation, arguments)
        tools.append({"operation": operation, "arguments": arguments, "result": result,
                      "sequence": session._environment._last_sequence, "phase": phase})
        return result

    async def resume():
        prior = session._status
        arguments = {"owner_id": "planner", "execution_id": prior["execution_id"],
                     "boundary_id": prior["boundary_event_id"], "state_version": prior["state_version"]}
        result = await session.resume(arguments)
        tools.append({"operation": "execution.resume", "arguments": arguments,
                      "result": result, "sequence": session._environment._last_sequence, "phase": phase})
        return await wait_boundary(session)

    async def snapshot(label):
        state = await session._device.on_owner(session._environment._backend().observe_control)
        state["label"] = label
        (args.output / f"sensor-{label}.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        (args.output / f"head-{label}.png").write_bytes(base64.b64decode(state["measurements"]["rgb_png_base64"], validate=True))
        return state

    initialized = False
    try:
        edh = root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd"
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
                                  "native_task_id": "office-proximity", "policy_id": "official-microduck-onnx",
                                  "execution_mode": "policy", "schema_path": str(edh / "harness/contracts/schema/physical.schema.json"),
                                  "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        initialized = True
        await session.open_task({"native_task_id": "office-proximity", "run_task_id": task_id,
                                 "catalog_task_id": "office-proximity"})
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id,
                   "team_run_id": task_id, "goal_id": "proximity", "attempt_id": "attempt-1",
                   "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
                   "required_capabilities": ["policy-navigation"],
                   "success_contract": {"id": "office-proximity", "version": "1",
                                        "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                                        "source": {"kind": "benchmark", "reference": "office-native-gt"}},
                   "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
                   "owner_assignment_id": "proximity-acceptance", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "office-proximity",
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
        await wait_boundary(session)
        await call("select_policy", {"policy_name": "alpha_walking"})
        initial = await snapshot("initial")
        expected_wall = configuration["expected_wall"]["path"].removeprefix("/Root/")
        measured_initial = initial["measurements"]
        wall_ray_count = sum(status == 5 and hit is not None and expected_wall in hit
                             for status, hit in zip(measured_initial["tof_status"],
                                                    measured_initial["tof_hit_geoms"], strict=True))
        if wall_ray_count == 0:
            raise AssertionError("Initial valid ToF rays do not hit the configured original wall")
        initial_central = [measured_initial["tof_distance_mm"][row*8+col]
                           for row in range(2, 6) for col in range(2, 6)
                           if measured_initial["tof_status"][row*8+col] == 5]
        if not initial_central or min(initial_central) < 90:
            raise AssertionError("Proximity acceptance must begin above the 90 mm threshold")
        await call("read_sensor", {"sensor": "tof"})
        phase = "approach"
        for _ in range(configuration["segment_limit"]):
            await call("set_command", {"command": {"twist": configuration["twist"]}, "max_control_steps": 100})
            await resume()
            progress = await call("progress", {})
            boundaries.append(progress)
            reason = progress["motion_guard"]["reason"]
            print(json.dumps({"phase": phase, "sequence": progress["sequence"], "guard": reason,
                              "closest_mm": progress["motion_guard"]["central_tof_closest_mm"],
                              "position": progress["body_position_m"]}), flush=True)
            if progress["contact_evidence"]["non_ground_external_contact_samples_total"]:
                raise AssertionError("External contact occurred during approach")
            if reason == "forward_proximity":
                break
            if reason != "command_segment_complete":
                raise AssertionError(f"Unexpected physical boundary: {reason}")
        if not boundaries or boundaries[-1]["motion_guard"]["reason"] != "forward_proximity":
            raise AssertionError("ToF proximity boundary was not reached")
        guard_state = await snapshot("guard")
        if not any(status == 5 and hit is not None and expected_wall in hit
                   for status, hit in zip(guard_state["measurements"]["tof_status"],
                                          guard_state["measurements"]["tof_hit_geoms"], strict=True)):
            raise AssertionError("Guard snapshot does not observe the configured original wall")
        phase = "stop"
        await call("set_command", {"command": {"twist": [0, 0, 0]}, "max_control_steps": 100})
        await resume()
        progress = await call("progress", {})
        boundary = session._gate.snapshot()
        finish = await call("finish_policy", {key: boundary[key] for key in ("execution_id", "generation", "boundary_id")})
        final = await snapshot("final")
        backend = session._environment._backend()
        facts = await session._device.on_owner(lambda: {
            "solver_type": type(backend.sim.solver).__name__, "solver_device": str(backend.sim.wp_data.qpos.device),
            "policy_sha256": backend.active_policy.sha256, "environment_geoms": len(backend._environment_geom_ids),
            "robot_geoms": len(backend._robot_geom_ids), "physics_calls": backend.native.scene["robot"].actuators["official_bam"].physics_calls})
        if (finish["execution"]["state"] != "ended" or finish["stop_confirmation"]["stopped_samples"] < 5
                or progress["fallen"] or final["measurements"]["height_m"] < .09
                or final["measurements"]["tilt_rad"] > .436332313
                or progress["contact_evidence"]["non_ground_external_contact_samples_total"]):
            raise AssertionError("Physical stop acceptance failed")
        result = {"accepted": True, "run_task_id": task_id, "configuration": {k: v for k, v in configuration.items() if k != "control_secret"},
                  "initial_position_m": initial["measurements"]["body_position_m"], "initial_wall_valid_rays": wall_ray_count,
                  "initial_central_tof_closest_mm": min(initial_central),
                  "guard": boundaries[-1]["motion_guard"], "final_position_m": final["measurements"]["body_position_m"],
                  "finish": finish, "boundaries": boundaries, "runtime": facts,
                  "scope": "Native execution and ToF proximity acceptance; VLM and formal navigation verdict remain pending",
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"accepted": True, "guard_sequence": result["guard"]["sequence"], "runtime": facts}), flush=True)
    finally:
        for name, values in (("samples", samples), ("frames", frames), ("tools", tools)):
            (args.output / f"{name}.json").write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
        if initialized:
            await session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
