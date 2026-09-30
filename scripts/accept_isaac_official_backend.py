import argparse
import base64
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend
from oh_my_duck.core.paths import project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="alpha_walking")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--lateral-velocity", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.scene_config.read_text())
    for key in ("usd_path", "provenance_path", "public_map_path"):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str(project_root() / config[key])
    backend = IsaacNewtonBamBackend(
        robot_id="isaac-official-acceptance", catalog_dir=args.catalog,
        scene_path=Path(config["usd_path"]), device=config["device"],
        scene_id=config["scene_id"], provenance_path=Path(config["provenance_path"]),
        public_map_path=Path(config["public_map_path"]) if config.get("public_map_path") else None)
    started = time.monotonic()
    writer = imageio.get_writer(args.output / "observer.mp4", fps=25)
    trajectory = []
    stop_trajectory = []
    try:
        state = backend.reset_episode(config.get("seed", 0), config["goal"], config["spawn_pose"])
        print(json.dumps({"phase": "initialized", "solver": type(backend.sim.solver).__name__,
                          "device": str(backend.sim.wp_data.qpos.device),
                          "environment_geoms": len(backend._environment_geom_ids),
                          "robot_geoms": len(backend._robot_geom_ids)}), flush=True)
        (args.output / "head_initial.png").write_bytes(base64.b64decode(state["measurements"]["rgb_png_base64"]))
        spawn_position = state["measurements"]["body_position_m"]
        for index in range(300):
            inference = backend.infer_policy()
            state = backend.apply_policy_action(inference["action"], f"settle:{index}")
            if backend._stopped_samples >= 5:
                break
        backend.select_policy(args.policy, "select-policy")
        initial = state["measurements"]["body_position_m"]
        command = [args.velocity, args.lateral_velocity, 0]
        if args.policy in {"alpha_walking", "velstand"}:
            backend.set_command({"twist": command}, "move-command")
        for index in range(args.steps):
            inference = backend.infer_policy()
            if inference["complete"]:
                break
            state = backend.apply_policy_action(inference["action"], f"action:{index}")
            measured = state["measurements"]
            if (not np.isfinite(measured["policy_observation"]).all()
                    or len(measured["policy_observation"]) != 61
                    or len(measured["joint_position_rad"]) != 14 or state["raw_sim_steps"] != 4):
                raise RuntimeError("Newton action or observation dimensions/timing differ")
            trajectory.append({"sequence": state["sequence"], "simulation_time_s": state["simulation_time_s"],
                               "position_m": measured["body_position_m"], "tilt_rad": measured["tilt_rad"],
                               "body_twist": measured["body_twist"], "policy_action": inference["action"],
                               "raw_sim_steps": state["raw_sim_steps"],
                               "contact_evidence": measured["contact_evidence"]})
            if index % 2 == 0:
                frame = backend.capture_observer()
                writer.append_data(imageio.imread(base64.b64decode(frame["rgb_png_base64"])))
            if args.policy == "alpha_walking" and measured["fallen"]:
                raise RuntimeError("Walking policy lost upright state")
            if index % 50 == 0:
                print(json.dumps({"phase": "motion", "step": index,
                                  "position_m": measured["body_position_m"],
                                  "body_twist": measured["body_twist"]}), flush=True)
        backend.set_command({"twist": [0, 0, 0]}, "stop-command")
        for index in range(300):
            inference = backend.infer_policy()
            if inference["complete"]:
                raise RuntimeError("Policy duration ended before a measured stop")
            state = backend.apply_policy_action(inference["action"], f"stop:{index}")
            measured = state["measurements"]
            stop_trajectory.append({"sequence": state["sequence"],
                                    "simulation_time_s": state["simulation_time_s"],
                                    "body_twist": measured["body_twist"],
                                    "stopped_samples": backend._stopped_samples})
            if backend._stopped_samples >= 5:
                break
        if backend._stopped_samples < 5:
            raise RuntimeError("Zero command did not produce five measured stopped samples")
        backend.signal_stop()
        final = state["measurements"]
        if final["fallen"] or final["height_m"] < 0.09 or final["tilt_rad"] > math.radians(25):
            raise RuntimeError("Measured stop did not preserve upright state")
        displacement = float(np.linalg.norm(np.asarray(final["body_position_m"])[:2] - np.asarray(initial)[:2]))
        if displacement < 0.02:
            raise RuntimeError(f"Commanded motion displacement is below 0.02 m: {displacement}")
        report = {"scene": backend.public_scene_info(), "policy": args.policy,
                  "policy_sha256": backend.active_policy.sha256,
                  "initial_position_m": initial, "final_position_m": final["body_position_m"],
                  "spawn_position_m": spawn_position, "motion_command": command,
                  "displacement_xy_m": displacement, "control_steps": len(trajectory),
                  "physics_calls": backend.native.scene["robot"].actuators["official_bam"].physics_calls,
                  "solver_type": type(backend.sim.solver).__name__, "solver_device": str(backend.sim.wp_data.qpos.device),
                  "confirmed_stopped": backend._stopped_samples >= 5,
                  "stop_control_steps": len(stop_trajectory), "stop_final_twist": final["body_twist"],
                  "wall_time_s": time.monotonic() - started,
                  "final_state": state, "goal": backend.check_goal(),
                  "validation_scope": "Real Newton GPU physics and official policy actions; native Harness acceptance separate"}
        (args.output / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
        (args.output / "stop-trajectory.json").write_text(json.dumps(stop_trajectory, indent=2) + "\n")
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "head_final.png").write_bytes(base64.b64decode(final["rgb_png_base64"]))
        print(json.dumps({"policy": args.policy, "steps": len(trajectory), "displacement_xy_m": displacement,
                          "final_height_m": final["height_m"], "final_tilt_rad": final["tilt_rad"]}))
    finally:
        (args.output / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
        (args.output / "stop-trajectory.json").write_text(json.dumps(stop_trajectory, indent=2) + "\n")
        writer.close()
        backend.close()


if __name__ == "__main__":
    main()
