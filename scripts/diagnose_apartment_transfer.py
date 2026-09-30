import argparse
from collections import deque
import json
import math
from pathlib import Path

from diagnose_apartment_warmup import move, standing_state, start_backend
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def ready(states: deque[dict]) -> bool:
    if len(states) < 25:
        return False
    if any(math.hypot(*state["body_twist"][:2]) >= 0.005 or
           abs(state["body_twist"][2]) >= 0.01 or
           state["tilt_rad"] >= 0.02 or
           max(abs(value) for value in state["joint_velocity_rad_s"]) >= 0.001
           for state in states):
        return False
    for joint in range(14):
        angles = [state["joint_position_rad"][joint] for state in states]
        if max(angles) - min(angles) >= 0.001:
            return False
    return True


def measured_entry(catalog: Path) -> dict:
    backend = CpuMujocoBamBackend(robot_id="measured-entry-diagnosis", catalog_dir=catalog)
    snapshots = []
    window = deque(maxlen=25)
    try:
        first = backend.reset_episode(seed=20260929,
                                      goal={"kind": "room", "room": "office", "hold_ticks": 5})
        window.append(standing_state(backend, first["measurements"]))
        for index in range(300):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(
                inference["action"], request_id=f"measured-entry:velstand:{index}",
                expected_sequence=inference["sequence"], should_stop=lambda: False)
            state = standing_state(backend, result["measurements"])
            window.append(state)
            if backend._sequence % 10 == 0 or ready(window):
                snapshots.append(state)
            if ready(window):
                break
        else:
            raise RuntimeError("Measured standing-entry condition was not reached")
        entry_sequence = backend._sequence
        backend.select_policy("alpha_walking", request_id="measured-entry:select")
        approach = move(backend, "measured-entry", "approach", [0.2, 0.25, 0], 800)
    finally:
        backend.close()
    return {"entry_sequence": entry_sequence, "entry_rule": {
        "window_control_steps": 25, "body_linear_speed_limit_m_s": 0.005,
        "body_yaw_speed_limit_rad_s": 0.01, "tilt_limit_rad": 0.02,
        "per_joint_speed_limit_rad_s": 0.001,
        "per_joint_position_range_limit_rad": 0.001},
        "standing_snapshots": snapshots, "approach": approach}


def recovered_entry(catalog: Path) -> dict:
    backend, setup = start_backend("recovered-entry", catalog, 75)
    try:
        approach = move(backend, "recovered-entry", "approach", [0.2, 0.25, 0], 800)
        if approach["stop_reason"] != "forward_tof_stop":
            raise RuntimeError("Recovery diagnosis requires the measured ToF stop")
        recovery = move(backend, "recovered-entry", "recovery", [-0.2, 0.3, 0], 150)
        if recovery["last"]["contact_evidence"]["non_ground_external_contact_samples_total"]:
            raise RuntimeError("Recovery path contacted a native obstacle")
        resumed = move(backend, "recovered-entry", "resumed", [0.2, 0.25, 0], 700)
    finally:
        backend.close()
    return {"setup": setup, "phases": [approach, recovery, resumed]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalog = Path(__file__).resolve().parents[1] / ".cache" / "official-policies" / OFFICIAL_REVISION
    output = {"seed": 20260929, "spawn_pose": [0, 0, 0],
              "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
              "measured_entry": measured_entry(catalog),
              "recovered_entry": recovered_entry(catalog)}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"measured_entry_sequence": output["measured_entry"]["entry_sequence"],
                      "measured_approach": {key: output["measured_entry"]["approach"][key]
                                            for key in ("executed_controls", "stop_reason", "last")},
                      "recovered_phases": [{key: phase[key] for key in
                                             ("phase", "executed_controls", "stop_reason", "last")}
                                            for phase in output["recovered_entry"]["phases"]]}, indent=2))


if __name__ == "__main__":
    main()
