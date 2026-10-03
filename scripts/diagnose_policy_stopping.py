import argparse
import json
import math
from pathlib import Path

from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--drive-steps", type=int, default=50)
    parser.add_argument("--brake-steps", type=int, default=50)
    args = parser.parse_args()
    if not -0.4 <= args.brake <= 0 or args.drive_steps < 1 or args.brake_steps < 1:
        raise ValueError("Invalid stopping diagnosis parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.scene_config.read_text())
    root = Path(__file__).resolve().parents[1]
    backend = IsaacNewtonBamBackend(robot_id="stopping-diagnosis", catalog_dir=args.catalog,
        scene_path=root / config["usd_path"], device=config["device"], scene_id=config["scene_id"],
        provenance_path=root / config["provenance_path"], public_map_path=root / config["public_map_path"],
        robot_model=config.get("robot_model", "allcollisions"))
    rows = []
    try:
        backend.reset_episode(20261003, config["goal"], config["spawn_pose"])
        locomotion = "roller" if backend.robot_model == "groundcontact_rollers" else "alpha_walking"
        for phase, velocity, steps in (("warmup", 0, 225), ("drive", 0.4, args.drive_steps),
                                       ("brake", args.brake, args.brake_steps), ("settle", 0, 175)):
            if phase == "drive" and backend.active_policy.name != locomotion:
                backend.select_policy(locomotion, "diagnosis:select")
            backend.set_command({"twist": [velocity, 0, 0]}, phase + ":command")
            for index in range(steps):
                inference = backend.infer_policy()
                if inference["complete"]:
                    raise AssertionError("Continuous policy terminated")
                result = backend.apply_policy_action(inference["action"], f"{phase}:{index}")
                measured = result["measurements"]
                if len(measured["policy_observation"]) != 61 or len(inference["action"]) != 14 or result["raw_sim_steps"] != 4:
                    raise AssertionError("Canonical policy timing or dimensions differ")
                row = {key: measured[key] for key in ("body_position_m", "body_twist", "body_twist_world",
                    "height_m", "tilt_rad", "odometry", "passive_joints", "contact_evidence")}
                row.update(phase=phase, sequence=result["sequence"], command_velocity_m_s=velocity,
                           stopped_samples=backend._stopped_samples)
                rows.append(row)
                if measured["fallen"] or measured["contact_evidence"]["current_control_non_ground_external"]:
                    raise AssertionError("Physical diagnosis encountered a fall or obstacle contact")
        drive = [row for row in rows if row["phase"] == "drive"]
        braking = [row for row in rows if row["phase"] in {"brake", "settle"}]
        start, final = drive[-1], rows[-1]
        summary = {"scope": "actual Newton/BAM stopping response; independent task verification separate",
            "robot_model": backend.robot_model, "drive_steps": args.drive_steps, "brake_velocity_m_s": args.brake,
            "drive_final_speed_m_s": start["body_twist"][0],
            "post_drive_translation_m": math.dist(start["body_position_m"][:2], final["body_position_m"][:2]),
            "first_stopped_sequence": next((row["sequence"] for row in braking if row["stopped_samples"] >= 5), None),
            "final_stopped_samples": final["stopped_samples"], "final_height_m": final["height_m"],
            "final_tilt_rad": final["tilt_rad"], "control_steps": len(rows), "raw_sim_steps": len(rows) * 4}
        (args.output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, allow_nan=False), flush=True)
    finally:
        (args.output / "samples.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
        backend.close()


if __name__ == "__main__":
    main()
