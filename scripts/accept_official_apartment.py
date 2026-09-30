import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.rl.evaluation.protocols import EvaluationProtocol
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend, MotionBusyError
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def run(output: Path, stand_ticks: int, move_ticks: int, stop_ticks: int,
        forward_speed: float, policy_name: str) -> dict:
    if min(stand_ticks, move_ticks, stop_ticks) < 1 or forward_speed <= 0:
        raise ValueError("Acceptance stages require positive ticks and forward speed")
    root = Path(__file__).resolve().parents[1]
    catalog = root / ".cache" / "official-policies" / OFFICIAL_REVISION
    backend = CpuMujocoBamBackend(robot_id="official-apartment-acceptance", catalog_dir=catalog)
    try:
        initial = backend.reset_episode(seed=42, goal={"kind": "room", "room": "corridor", "hold_ticks": 5})
        prelude_actions = 0
        if policy_name != "velstand":
            for index in range(50):
                inference = backend.infer_policy()
                backend.apply_policy_action(inference["action"], request_id=f"prelude:{index}",
                                            expected_sequence=inference["sequence"],
                                            should_stop=lambda: False)
                prelude_actions += 1
            backend.select_policy(policy_name, request_id=f"select:{policy_name}")
        spec = backend.action_spec()
        png = base64.b64decode(initial["measurements"]["rgb_png_base64"])
        image = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
        if image.shape != (240, 320, 3) or np.ptp(image) < 1:
            raise ValueError("Initial native head camera PNG has invalid dimensions or pixels")
        (output / "head_camera_initial.png").write_bytes(png)
        trace = []
        admitted = prelude_actions
        raw_steps = 4 * prelude_actions
        for name, count, twist in (
            ("stand", stand_ticks, [0.0, 0.0, 0.0]),
            ("move", move_ticks, [forward_speed, 0.0, 0.0]),
            ("stop", stop_ticks, [0.0, 0.0, 0.0]),
        ):
            backend.set_command({"twist": twist}, request_id=f"command:{name}")
            for index in range(count):
                inference = backend.infer_policy()
                action = np.asarray(inference["action"])
                lower, upper = np.asarray(spec["minimum"]), np.asarray(spec["maximum"])
                if (action < lower).any() or (action > upper).any():
                    raise ValueError("Official action exceeded source position-target range")
                result = backend.apply_policy_action(inference["action"], request_id=f"{name}:{index}",
                                                     expected_sequence=inference["sequence"],
                                                     should_stop=lambda: False)
                admitted += 1
                raw_steps += result["raw_sim_steps"]
                state = result["measurements"]
                trace.append({"stage": name, "sequence": result["sequence"],
                              "simulation_time_s": result["simulation_time_s"],
                              "position_m": state["body_position_m"],
                              "twist": state["body_twist"],
                              "twist_world": state["body_twist_world"],
                              "observed_command": inference["observation"][48:61],
                              "tilt_rad": state["tilt_rad"],
                              "height_m": state["height_m"], "fallen": state["fallen"],
                              "action_min": float(action.min()), "action_max": float(action.max()),
                              "tof_valid": state["tof_status"].count(5),
                              "tof_no_target": state["tof_status"].count(255)})
                if result["interrupted"] or result["raw_sim_steps"] != 4:
                    raise RuntimeError("An admitted action did not execute four native physics steps")
        if not trace:
            raise ValueError("At least one action is required")
        before = backend.controller.q_target.copy()
        pending = backend.infer_policy()
        rejected = False
        try:
            backend.apply_policy_action(pending["action"], request_id="rejected-after-stop",
                                        expected_sequence=pending["sequence"], should_stop=lambda: True)
        except MotionBusyError:
            rejected = True
        if not rejected or not np.array_equal(before, backend.controller.q_target):
            raise ValueError("Stop boundary admitted an action or changed BAM targets")
        goal = backend.check_goal()
        first = trace[0]["position_m"]
        move = [item for item in trace if item["stage"] == "move"]
        stopped = [item for item in trace if item["stage"] == "stop"]
        displacement = float(move[-1]["position_m"][0] - move[0]["position_m"][0]) if move else 0.0
        move_duration_s = move_ticks * spec["physics_dt_s"] * spec["physics_steps_per_action"]
        commanded_displacement_m = forward_speed * move_duration_s
        move_tracking_ratio = displacement / commanded_displacement_m
        stop_displacement_m = float(stopped[-1]["position_m"][0] - stopped[0]["position_m"][0])
        stop_drift_ratio = abs(stop_displacement_m) / abs(displacement) if displacement else float("inf")
        final_twist = stopped[-1]["twist"] if stopped else trace[-1]["twist"]
        stable = all(not item["fallen"] for item in trace)
        stop_confirmed = backend._stopped_samples >= 5
        tracking_passed = 0.5 <= move_tracking_ratio <= 1.5
        stop_passed = stop_drift_ratio <= 0.1 and max(abs(value) for value in final_twist) <= 0.1
        protocol = EvaluationProtocol((), "walking", 0.065, np.pi / 3)
        score = protocol.score({"height": [item["height_m"] for item in trace],
                                "tilt": [item["tilt_rad"] for item in trace],
                                "twist": [item["twist"] for item in trace],
                                "command": [[0.0, 0.0, 0.0] if item["stage"] != "move"
                                            else [forward_speed, 0.0, 0.0] for item in trace]},
                               completed=True)
        passed = bool(stable and tracking_passed and stop_passed and stop_confirmed
                      and rejected and score["success"])
        return {"status": "passed" if passed else "behavior_failed", "policy": policy_name,
                "policy_sha256": spec["policy_sha256"], "scene": "official scene_apartment.xml",
                "actuator": spec["actuator"], "prelude_action_count": prelude_actions,
                "admitted_action_count": admitted,
                "raw_physics_steps": raw_steps, "actions_after_stop": 0,
                "action_min": min(item["action_min"] for item in trace),
                "action_max": max(item["action_max"] for item in trace),
                "start_position_m": first, "final_position_m": trace[-1]["position_m"],
                "move_forward_displacement_m": displacement,
                "move_commanded_displacement_m": commanded_displacement_m,
                "move_tracking_ratio": move_tracking_ratio,
                "move_tracking_passed": tracking_passed,
                "stop_forward_displacement_m": stop_displacement_m,
                "stop_drift_ratio": stop_drift_ratio,
                "stop_passed": stop_passed,
                "final_body_twist": final_twist,
                "final_body_twist_world": stopped[-1]["twist_world"],
                "maximum_tilt_rad": max(item["tilt_rad"] for item in trace),
                "minimum_height_m": min(item["height_m"] for item in trace),
                "stop_confirmed_samples": backend._stopped_samples,
                "camera": {"format": "PNG", "width": 320, "height": 240,
                           "pixel_min": int(image.min()), "pixel_max": int(image.max()),
                           "pixel_variance": float(image.var())},
                "tof": {"valid_hits_last": trace[-1]["tof_valid"],
                        "no_target_last": trace[-1]["tof_no_target"]},
                "native_goal": goal, "action_spec": spec, "protocol_score": score,
                "trace": trace}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stand-ticks", type=int, default=50)
    parser.add_argument("--move-ticks", type=int, default=100)
    parser.add_argument("--stop-ticks", type=int, default=100)
    parser.add_argument("--forward-speed", type=float, default=0.3)
    parser.add_argument("--policy", choices=("velstand", "alpha_walking"), default="velstand")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.output, args.stand_ticks, args.move_ticks, args.stop_ticks,
                 args.forward_speed, args.policy)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"trace", "action_spec", "native_goal"}}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
