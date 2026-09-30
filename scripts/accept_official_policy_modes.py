import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
from oh_my_duck.robotics.microduck.protocol import HOME


POLICY_TICKS = {
    "alpha_stand": 200,
    "sitstand": 200,
    "ground_pick": 140,
    "roulade": 50,
    "kick_left": 25,
    "kick_right": 25,
    "roller": 0,
    "crouch": 0,
}


def run(policy_name: str, output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    catalog = root / ".cache" / "official-policies" / OFFICIAL_REVISION
    backend = CpuMujocoBamBackend(robot_id=f"official-{policy_name}", catalog_dir=catalog)
    try:
        inventory = backend.list_policies()
        if len(inventory["policies"]) != 10 or any(
            not policy["apartment_behavior_status"] for policy in inventory["policies"]
        ):
            raise ValueError("Official apartment policy inventory is incomplete")
        initial = backend.reset_episode(seed=42, goal={"kind": "room", "room": "corridor", "hold_ticks": 5})
        png = base64.b64decode(initial["measurements"]["rgb_png_base64"])
        image = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
        if image.shape != (240, 320, 3) or np.ptp(image) < 1:
            raise ValueError("Native head camera PNG failed format or pixel validation")
        (output / "head_camera_initial.png").write_bytes(png)
        prelude = []
        for index in range(50):
            inference = backend.infer_policy()
            if inference["complete"]:
                raise ValueError("Default walk policy completed during the standing prelude")
            result = backend.apply_policy_action(inference["action"], request_id=f"prelude:{index}",
                                                 expected_sequence=inference["sequence"],
                                                 should_stop=lambda: False)
            prelude.append(result["raw_sim_steps"])
        if sum(prelude) != 200:
            raise ValueError("Standing prelude did not execute 200 MuJoCo steps")
        prelude_end = result["measurements"]
        if policy_name in {"roller", "crouch"}:
            rejected = False
            try:
                backend.select_policy(policy_name, request_id=f"select:{policy_name}")
            except ValueError as error:
                if "roller robot model" not in str(error):
                    raise
                rejected = True
            if not rejected:
                raise ValueError("Roller policy was admitted on the wheel-less apartment robot")
            return {"policy": policy_name, "status": "requires_roller_robot_model",
                    "admitted_action_count": 50, "raw_physics_steps": 200,
                    "rejected_before_policy_action": True, "inventory": inventory}
        selected = backend.select_policy(policy_name, request_id=f"select:{policy_name}")
        if selected["action_spec"]["policy_name"] != policy_name:
            raise ValueError("Official policy selection returned another model")
        trace = []
        for index in range(POLICY_TICKS[policy_name]):
            if policy_name == "alpha_stand" and index in {0, 100}:
                backend.set_command(
                    {"head": [0.0, 0.2 if index == 0 else 0.0, 0.0, 0.0],
                     "body": [0.0, 0.0, 0.01 if index == 0 else 0.0, 0.0, 0.0, 0.0]},
                    request_id=f"pose:{index}")
            if policy_name == "sitstand" and index in {0, 100}:
                backend.set_command({"posture": "sit" if index == 0 else "stand"},
                                    request_id=f"posture:{index}")
            inference = backend.infer_policy()
            if inference["complete"]:
                raise ValueError("Official episodic policy completed before manifest duration")
            action = np.asarray(inference["action"])
            spec = selected["action_spec"]
            if (action < np.asarray(spec["minimum"])).any() or (action > np.asarray(spec["maximum"])).any():
                raise ValueError("Official policy action exceeded source actuator target range")
            result = backend.apply_policy_action(inference["action"], request_id=f"mode:{index}",
                                                 expected_sequence=inference["sequence"],
                                                 should_stop=lambda: False)
            if result["raw_sim_steps"] != 4 or result["interrupted"]:
                raise ValueError("Official action did not execute four physics substeps")
            state = result["measurements"]
            trace.append({"sequence": result["sequence"], "time_s": result["simulation_time_s"],
                          "command": inference["command_block"], "position_m": state["body_position_m"],
                          "body_twist": state["body_twist"], "height_m": state["height_m"],
                          "head_pitch_rad": state["joint_position_rad"][6],
                          "tilt_rad": state["tilt_rad"], "fallen": state["fallen"],
                          "episode_terminated": state["episode_terminated"],
                          "episode_termination_reason": state["episode_termination_reason"],
                          "action_min": float(action.min()), "action_max": float(action.max()),
                          "tof_valid_hits": state["tof_status"].count(5),
                          "tof_no_target": state["tof_status"].count(255)})
        completion = backend.infer_policy() if selected["duration_s"] is not None else None
        if completion is not None and (not completion["complete"] or "action" in completion):
            raise ValueError("Manifest duration failed to end policy actions")
        if selected["duration_s"] is None and trace[-1]["episode_terminated"]:
            raise ValueError("Perpetual or scripted policy terminated unexpectedly")
        return {"policy": policy_name, "sha256": selected["sha256"],
                "kind": selected["kind"], "encoding": selected["encoding"],
                "duration_s": selected["duration_s"], "selection": selected,
                "inventory": inventory,
                "status": "execution_checked_behavior_unverified", "prelude_action_count": 50,
                "admitted_action_count": 50 + len(trace), "raw_physics_steps": 200 + 4 * len(trace),
                "actions_after_completion": 0 if completion is not None else None,
                "completion": completion, "minimum_height_m": min(item["height_m"] for item in trace),
                "maximum_tilt_rad": max(item["tilt_rad"] for item in trace),
                "final_position_m": trace[-1]["position_m"],
                "final_body_twist": trace[-1]["body_twist"],
                "baseline_height_m": prelude_end["height_m"],
                "baseline_head_pitch_rad": prelude_end["joint_position_rad"][6],
                "head_pitch_home_rad": HOME[6],
                "camera": {"format": "PNG", "width": 320, "height": 240,
                           "pixel_min": int(image.min()), "pixel_max": int(image.max()),
                           "pixel_variance": float(image.var())},
                "trace": trace}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=tuple(POLICY_TICKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.policy, args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"trace", "selection", "inventory"}}, indent=2))


if __name__ == "__main__":
    main()
