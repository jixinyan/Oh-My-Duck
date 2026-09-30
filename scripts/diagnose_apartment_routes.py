import argparse
import json
from pathlib import Path

from diagnose_apartment_collision import snapshot
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


TRIALS = (
    ("diagonal", (("diagonal", 300, [0.2, 0.25, 0.0]),)),
    ("side_then_forward", (("side", 180, [0.0, 0.25, 0.0]),
                           ("forward", 250, [0.3, 0.0, 0.0]))),
    ("turn_then_forward", (("turn", 100, [0.0, 0.0, 0.7]),
                           ("forward", 250, [0.25, 0.0, 0.0]))),
    ("side_only", (("side", 200, [0.0, 0.25, 0.0]),)),
)


def run_trial(name: str, phases: tuple, catalog_dir: Path) -> dict:
    backend = CpuMujocoBamBackend(robot_id=f"route-diagnosis-{name}", catalog_dir=catalog_dir)
    traces = []
    checkpoints = []
    try:
        first = backend.reset_episode(seed=20260929,
                                      goal={"kind": "room", "room": "office", "hold_ticks": 5})
        traces.append(snapshot(backend, first["measurements"], "initial"))
        for index in range(187):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(
                inference["action"], request_id=f"{name}:velstand:{index}",
                expected_sequence=inference["sequence"], should_stop=lambda: False)
        traces.append(snapshot(backend, result["measurements"], "velstand_ready"))
        backend.select_policy("alpha_walking", request_id=f"{name}:select")
        for phase, controls, command in phases:
            start = backend._native_state()
            start_sequence = backend._sequence
            backend.set_command({"twist": command}, request_id=f"{name}:command:{phase}")
            for index in range(controls):
                inference = backend.infer_policy()
                result = backend.apply_policy_action(
                    inference["action"], request_id=f"{name}:{phase}:{index}",
                    expected_sequence=inference["sequence"], should_stop=lambda: False)
                if result["sequence"] % 25 == 0 or index == controls - 1:
                    traces.append(snapshot(backend, result["measurements"], phase))
            end = result["measurements"]
            checkpoints.append({"phase": phase, "command": command,
                                "start_sequence": start_sequence,
                                "end_sequence": backend._sequence,
                                "start_position_m": start["body_position_m"],
                                "end_position_m": end["body_position_m"],
                                "start_yaw_rad": start["odometry"]["yaw_rad"],
                                "end_yaw_rad": end["odometry"]["yaw_rad"],
                                "end_fallen": end["fallen"]})
        goal = backend.check_goal()
    finally:
        backend.close()
    return {"trial": name, "seed": 20260929,
            "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
            "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
            "phases": checkpoints, "goal_check": goal, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    catalog = root / ".cache" / "official-policies" / OFFICIAL_REVISION
    results = [run_trial(name, phases, catalog) for name, phases in TRIALS]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{"trial": item["trial"], "phases": item["phases"],
                       "goal_reached": item["goal_check"]["checks"]["goal_reached"]["satisfied"]}
                      for item in results], indent=2))


if __name__ == "__main__":
    main()
