import argparse
import base64
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-roulade", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.scene_config.read_text())
    root = Path(__file__).resolve().parents[1]
    backend = IsaacNewtonBamBackend(robot_id="policy-showcase-acceptance", catalog_dir=args.catalog,
        scene_path=root / config["usd_path"], device=config["device"], scene_id=config["scene_id"],
        provenance_path=root / config["provenance_path"], public_map_path=root / config["public_map_path"],
        robot_model=config.get("robot_model", "allcollisions"), observer_renderer=config.get("observer_renderer", "newton_warp"))
    trace, phases = [], []
    writer = imageio.get_writer(args.output / "observer.mp4", fps=25)
    report = {"status": "running", "scope": "actual Newton/BAM policy effects and measurements",
              "trace": trace, "phases": phases}

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def phase(name, policy, command, ticks, transition=False):
        if backend.active_policy.name != policy:
            selection = (backend.transition_completed_policy if transition else backend.select_policy)(policy, name + ":select")
        else:
            selection = {"policy_name": policy, "sha256": backend.active_policy.sha256}
        backend.set_command(command, name + ":command")
        first = backend._native_state()
        start = len(trace)
        for index in range(ticks):
            inference = backend.infer_policy()
            if inference["complete"]:
                raise ValueError("Policy completed before its declared showcase phase")
            state = backend.apply_policy_action(inference["action"], f"{name}:{index}")
            measured = state["measurements"]
            if state["raw_sim_steps"] != 4 or len(measured["policy_observation"]) != 61 or len(inference["action"]) != 14:
                raise ValueError("Canonical observation/action/timing changed")
            row = {key: measured[key] for key in ("body_position_m", "body_twist", "height_m", "tilt_rad",
                "joint_position_rad", "joint_velocity_rad_s", "passive_joints", "contact_evidence")}
            row.update(sequence=state["sequence"], simulation_time_s=state["simulation_time_s"],
                       phase=name, policy_name=policy, action=inference["action"],
                       stopped_samples=backend._stopped_samples)
            if "mouth_tip" in backend.robot.site_names:
                row["mouth_tip_world_m"] = backend.robot.data.site_pos_w[0, backend.robot.site_names.index("mouth_tip")].cpu().tolist()
            trace.append(row)
            if index % 2 == 0:
                frame = backend.capture_observer()
                writer.append_data(imageio.imread(base64.b64decode(frame["rgb_png_base64"])))
        rows = trace[start:]
        current = rows[-1]
        phases.append({"name": name, "selection": selection, "command": command, "control_steps": ticks,
            "height_min_m": min(row["height_m"] for row in rows), "final_height_m": current["height_m"],
            "tilt_max_rad": max(row["tilt_rad"] for row in rows), "final_tilt_rad": current["tilt_rad"],
            "displacement_xy_m": float(np.linalg.norm(np.asarray(current["body_position_m"])[:2] - np.asarray(first["body_position_m"])[:2])),
            "stopped_samples": current["stopped_samples"], "sequence": current["sequence"]})
        save()
        print(json.dumps(phases[-1]), flush=True)

    try:
        backend.reset_episode(20260930, config["goal"], config["spawn_pose"])
        report["scene"] = backend.public_scene_info()
        zero = {"twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]}
        roller = backend.robot_model == "groundcontact_rollers"
        phase("warmup", "roller" if roller else "velstand", zero, 100)
        if roller:
            phase("roller-forward", "roller", {"twist": [0.2, 0, 0]}, 200)
            phase("roller-stop", "roller", zero, 150)
            phase("crouch-glide", "crouch", zero, 175)
            phase("roller-recovery", "roller", zero, 150, transition=True)
            wheel_names = [name for name in backend.robot.joint_names if name.startswith("passive_") and name.endswith("wheel")]
            if len(wheel_names) != 4 or phases[1]["displacement_xy_m"] < 0.1:
                raise AssertionError("Actual roller geometry or rolling displacement failed")
            if not any(abs(row["passive_joints"][name]["velocity_rad_s"]) > 0.1
                       for row in trace if row["phase"] == "roller-forward" for name in wheel_names):
                raise AssertionError("Passive wheels did not physically rotate")
        else:
            phase("sit", "sitstand", {**zero, "posture": "sit"}, 100)
            phase("stand", "sitstand", {**zero, "posture": "stand"}, 100)
            phase("head-body", "alpha_stand", {**zero, "head": [0, 0.2, 0.4, 0], "body": [0, 0, 0.01, 0, 0, 0]}, 100)
            phase("head-neutral", "alpha_stand", zero, 100)
            phase("ground-pick", "ground_pick", zero, 140)
            phase("pick-recovery", "alpha_stand", zero, 150, transition=True)
            if args.include_roulade:
                phase("roulade", "roulade", zero, 50)
                phase("roll-recovery", "alpha_stand", zero, 200, transition=True)
            if phases[1]["final_height_m"] >= phases[2]["final_height_m"] - 0.02:
                raise AssertionError("Measured sit/stand height difference is below 0.02 m")
        final = backend._native_state()
        if backend._stopped_samples < 5 or final["height_m"] < 0.09 or final["tilt_rad"] > np.deg2rad(25):
            raise AssertionError("Final policy recovery lacks an upright measured stop")
        report.update(status="passed", control_steps=len(trace), raw_sim_steps=4 * len(trace),
                      object_pick_status="not_supported_by_independent_jaw_actuation")
        save()
    finally:
        writer.close()
        backend.close()


if __name__ == "__main__":
    main()
