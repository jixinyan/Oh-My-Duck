import argparse
import json
from pathlib import Path

from diagnose_apartment_warmup import move, start_backend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


COMMANDS = ((0.2, 0.25, 0.4), (0.15, 0.3, 0.4), (0.2, 0.3, 0.7))


def settle(backend, name: str) -> dict:
    backend.set_command({"twist": [0, 0, 0]}, request_id=f"{name}:command:stop")
    traces = []
    for index in range(50):
        inference = backend.infer_policy()
        result = backend.apply_policy_action(
            inference["action"], request_id=f"{name}:stop:{index}",
            expected_sequence=inference["sequence"], should_stop=lambda: False)
        measurements = result["measurements"]
        if index in (0, 24, 49):
            traces.append({"sequence": result["sequence"],
                           "position_m": measurements["body_position_m"],
                           "yaw_rad": measurements["odometry"]["yaw_rad"],
                           "body_twist": measurements["body_twist"],
                           "tilt_rad": measurements["tilt_rad"],
                           "contact_evidence": measurements["contact_evidence"]})
    return {"control_steps": 50, "traces": traces, "goal_check": backend.check_goal()}


def trial(name: str, command: tuple[float, float, float], catalog: Path) -> dict:
    backend, setup = start_backend(name, catalog, 75)
    try:
        approach = move(backend, name, "approach", [0.2, 0.25, 0], 800)
        if approach["stop_reason"] != "forward_tof_stop":
            raise RuntimeError("The measured approach did not reach its ToF stop")
        recovery = move(backend, name, "recovery", [-0.2, 0.3, 0], 150)
        if recovery["stop_reason"] != "control_limit":
            raise RuntimeError("The measured recovery ended unexpectedly")
        resumed = move(backend, name, "resumed", list(command), 500)
        if resumed["stop_reason"] != "goal_reached":
            raise RuntimeError("The measured recovery did not reach office")
        stopped = settle(backend, name)
    finally:
        backend.close()
    return {"trial": name, "command": list(command), "setup": setup,
            "phases": [approach, recovery, resumed], "stopped": stopped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalog = Path(__file__).resolve().parents[1] / ".cache" / "official-policies" / OFFICIAL_REVISION
    trials = [trial(f"recovery-yaw-{index}", command, catalog)
              for index, command in enumerate(COMMANDS)]
    args.output.write_text(json.dumps(trials, indent=2) + "\n")
    print(json.dumps([{"command": item["command"],
                       "resumed": {key: item["phases"][-1][key] for key in
                                   ("executed_controls", "stop_reason", "last")},
                       "stopped": item["stopped"]}
                      for item in trials], indent=2))


if __name__ == "__main__":
    main()
