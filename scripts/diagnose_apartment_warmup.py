import argparse
import json
from pathlib import Path

from diagnose_apartment_diagonals import concise
from diagnose_apartment_collision import snapshot
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


WARMUP_CHECKPOINTS = {0, 25, 50, 75, 100, 125, 150, 175, 187}


def standing_state(backend: CpuMujocoBamBackend, measurements: dict) -> dict:
    return {"sequence": backend._sequence,
            "position_m": measurements["body_position_m"],
            "yaw_rad": measurements["odometry"]["yaw_rad"],
            "body_twist": measurements["body_twist"],
            "height_m": measurements["height_m"],
            "tilt_rad": measurements["tilt_rad"],
            "joint_position_rad": measurements["joint_position_rad"],
            "joint_velocity_rad_s": measurements["joint_velocity_rad_s"],
            "policy_action": measurements["policy_action"],
            "controller_targets_rad": measurements["controller_targets_rad"]}


def central_distance(measurements: dict) -> dict:
    zones = [row * 8 + col for row in range(2, 6) for col in range(2, 6)]
    valid = [measurements["tof_distance_mm"][zone] for zone in zones
             if measurements["tof_status"][zone] == 5]
    return {"valid_count": len(valid), "minimum_mm": min(valid) if valid else None}


def start_backend(name: str, catalog: Path, warmup: int) -> tuple[CpuMujocoBamBackend, dict]:
    backend = CpuMujocoBamBackend(robot_id=f"warmup-diagnosis-{name}", catalog_dir=catalog)
    first = backend.reset_episode(seed=20260929,
                                  goal={"kind": "room", "room": "office", "hold_ticks": 5})
    standing = [standing_state(backend, first["measurements"])]
    for index in range(warmup):
        inference = backend.infer_policy()
        result = backend.apply_policy_action(
            inference["action"], request_id=f"{name}:velstand:{index}",
            expected_sequence=inference["sequence"], should_stop=lambda: False)
        if backend._sequence in WARMUP_CHECKPOINTS:
            standing.append(standing_state(backend, result["measurements"]))
    backend.select_policy("alpha_walking", request_id=f"{name}:select")
    return backend, {"warmup_controls": warmup, "standing": standing}


def move(backend: CpuMujocoBamBackend, name: str, phase: str,
         command: list[float], limit: int) -> dict:
    traces = []
    stop_reason = "control_limit"
    for index in range(limit):
        if index % 100 == 0:
            backend.set_command({"twist": command},
                                request_id=f"{name}:command:{phase}:{index // 100}")
        inference = backend.infer_policy()
        result = backend.apply_policy_action(
            inference["action"], request_id=f"{name}:{phase}:{index}",
            expected_sequence=inference["sequence"], should_stop=lambda: False)
        measurements = result["measurements"]
        distance = central_distance(measurements)
        contact = measurements["contact_evidence"]
        goal = backend.check_goal()["checks"]["goal_reached"]["satisfied"]
        trace = {"sequence": result["sequence"], "position_m": measurements["body_position_m"],
                 "yaw_rad": measurements["odometry"]["yaw_rad"],
                 "body_twist": measurements["body_twist"],
                 "height_m": measurements["height_m"], "tilt_rad": measurements["tilt_rad"],
                 "central_4x4": distance, "contact_evidence": contact,
                 "goal_reached": goal, "fallen": measurements["fallen"]}
        traces.append(trace)
        if contact["current_control_non_ground_external"]:
            stop_reason = "non_ground_contact"
        elif command[0] > 0 and distance["minimum_mm"] is not None and distance["minimum_mm"] < 90:
            stop_reason = "forward_tof_stop"
        elif measurements["fallen"]:
            stop_reason = "fallen"
        elif goal:
            stop_reason = "goal_reached"
        if stop_reason != "control_limit":
            break
    return {"phase": phase, "command": command, "control_limit": limit,
            "executed_controls": len(traces), "stop_reason": stop_reason,
            "start_sequence": traces[0]["sequence"] - 1,
            "last": traces[-1], "traces": traces}


def trial(name: str, catalog: Path, command: list[float], warmup: int,
          followup: list[float] | None = None) -> dict:
    backend, setup = start_backend(name, catalog, warmup)
    try:
        first = move(backend, name, "approach", command, 800)
        phases = [first]
        if followup is not None:
            phases.append(move(backend, name, "recovery", followup, 150))
        final = concise(snapshot(backend, backend.observe_control()["measurements"], "final"))
    finally:
        backend.close()
    return {"trial": name, "seed": 20260929, "spawn_pose": [0, 0, 0],
            "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
            "setup": setup, "phases": phases, "final": final}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standing-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalog = Path(__file__).resolve().parents[1] / ".cache" / "official-policies" / OFFICIAL_REVISION
    if args.standing_only:
        backend, setup = start_backend("standing-reference", catalog, 187)
        backend.close()
        args.output.write_text(json.dumps(setup, indent=2) + "\n")
        print(json.dumps(setup, indent=2))
        return
    trials = [trial("lateral_01", catalog, [0.1, 0.3, 0], 75),
              trial("lateral_015", catalog, [0.15, 0.3, 0], 75),
              trial("recovery", catalog, [0.2, 0.25, 0], 75, [-0.2, 0.3, 0])]
    args.output.write_text(json.dumps(trials, indent=2) + "\n")
    print(json.dumps([{"trial": item["trial"], "standing": item["setup"]["standing"],
                       "phases": [{"phase": phase["phase"], "command": phase["command"],
                                   "executed_controls": phase["executed_controls"],
                                   "stop_reason": phase["stop_reason"],
                                   "last": phase["last"]} for phase in item["phases"]]}
                      for item in trials], indent=2))


if __name__ == "__main__":
    main()
