import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_apartment_collision import contact_evidence, snapshot
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


COMMANDS = ((0.2, 0.25, 0.0), (0.2, 0.3, 0.0), (0.3, 0.3, 0.0))
GROUND_GEOMS = {"floor_w", "floor_e", "floor_corridor_s", "floor_corridor_n",
                "floor_kitchen", "floor_bath", "rug_living", "rug_bedroom"}


def concise(snapshot_data: dict) -> dict:
    zones = snapshot_data["tof"]["zones"]
    valid = [zone["raw_distance_mm"] for zone in zones if zone["status"] == 5]
    central = [zone for zone in zones if zone["row"] in (3, 4)
               and zone["col"] in (3, 4)]
    return {"sequence": snapshot_data["sequence"],
            "simulation_time_s": snapshot_data["simulation_time_s"],
            "position_m": snapshot_data["body_position_m"],
            "yaw_rad": snapshot_data["odometry"]["yaw_rad"],
            "body_twist": snapshot_data["body_twist"],
            "body_twist_world": snapshot_data["body_twist_world"],
            "height_m": snapshot_data["height_m"],
            "tilt_rad": snapshot_data["tilt_rad"],
            "runtime_contact_evidence": snapshot_data["contact_evidence"],
            "non_ground_contacts": {
                key: value for key, value in
                snapshot_data["contacts"]["external_geom_counts"].items()
                if key not in GROUND_GEOMS},
            "strongest_external_contacts": snapshot_data["contacts"]["strongest_external_contacts"],
            "tof_category_counts": snapshot_data["tof"]["category_counts"],
            "tof_external_geoms": snapshot_data["tof"]["external_first_hit_geoms"],
            "tof_central": [{"zone": zone["zone"], "status": zone["status"],
                             "distance_mm": zone["raw_distance_mm"],
                             "first_hit_geom": zone["first_hit_geom"],
                             "category": zone["category"]} for zone in central],
            "tof_valid_quantiles_mm": np.quantile(valid, [0, 0.1, 0.5, 0.9, 1]).tolist()
            if valid else None}


def run_trial(command: tuple[float, float, float], catalog: Path) -> dict:
    name = "_".join(f"{component:g}" for component in command)
    backend = CpuMujocoBamBackend(robot_id=f"diagonal-diagnosis-{name}", catalog_dir=catalog)
    traces = []
    control_trace = []
    stop_reason = "control_limit"
    try:
        first = backend.reset_episode(seed=20260929,
                                      goal={"kind": "room", "room": "office", "hold_ticks": 5})
        traces.append(concise(snapshot(backend, first["measurements"], "initial")))
        for index in range(187):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(
                inference["action"], request_id=f"{name}:velstand:{index}",
                expected_sequence=inference["sequence"], should_stop=lambda: False)
        traces.append(concise(snapshot(backend, result["measurements"], "velstand_ready")))
        backend.select_policy("alpha_walking", request_id=f"{name}:select")
        backend.set_command({"twist": list(command)}, request_id=f"{name}:command")
        for index in range(600):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(
                inference["action"], request_id=f"{name}:diagonal:{index}",
                expected_sequence=inference["sequence"], should_stop=lambda: False)
            contacts = contact_evidence(backend)
            collision = any(geom not in GROUND_GEOMS
                            for geom in contacts["external_geom_counts"])
            goal = backend.check_goal()["checks"]["goal_reached"]["satisfied"]
            fallen = bool(result["measurements"]["fallen"])
            measurements = result["measurements"]
            central_indices = [row * 8 + col for row in range(2, 6) for col in range(2, 6)]
            central_valid = [measurements["tof_distance_mm"][zone]
                             for zone in central_indices
                             if measurements["tof_status"][zone] == 5]
            control_trace.append({
                "sequence": result["sequence"],
                "position_m": measurements["body_position_m"],
                "yaw_rad": measurements["odometry"]["yaw_rad"],
                "tilt_rad": measurements["tilt_rad"],
                "central_4x4_valid_count": len(central_valid),
                "central_4x4_min_mm": min(central_valid) if central_valid else None,
                "central_4x4_q10_mm": float(np.quantile(central_valid, 0.1))
                if central_valid else None,
                "central_4x4_median_mm": float(np.median(central_valid))
                if central_valid else None,
                "non_ground_contact_geoms": {
                    geom: count for geom, count in contacts["external_geom_counts"].items()
                    if geom not in GROUND_GEOMS},
                "goal_reached": goal, "fallen": fallen})
            if result["sequence"] % 25 == 0 or collision or goal or fallen or index == 599:
                traces.append(concise(snapshot(backend, result["measurements"], "diagonal")))
            if collision or goal or fallen:
                stop_reason = "non_ground_contact" if collision else "goal_reached" if goal else "fallen"
                break
    finally:
        backend.close()
    return {"command": list(command), "seed": 20260929,
            "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
            "walking_controls": index + 1, "stop_reason": stop_reason,
            "traces": traces, "control_trace": control_trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    catalog = root / ".cache" / "official-policies" / OFFICIAL_REVISION
    trials = [run_trial(command, catalog) for command in COMMANDS]
    args.output.write_text(json.dumps(trials, indent=2) + "\n")
    print(json.dumps([{"command": trial["command"],
                       "walking_controls": trial["walking_controls"],
                       "stop_reason": trial["stop_reason"],
                       "final": trial["traces"][-1]} for trial in trials], indent=2))


if __name__ == "__main__":
    main()
