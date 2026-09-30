import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
from oh_my_duck.robotics.microduck.sim_sensors import tof_directions


MILESTONES = {0, 187, 299, 497, 592, 725, 815, 967, 1080, 1205}


def contact_evidence(backend: CpuMujocoBamBackend) -> dict:
    model = backend.model
    data = backend.data
    robot_root = int(model.body_rootid[backend.policy_inference.trunk_base_id])
    external = []
    for index in range(data.ncon):
        contact = data.contact[index]
        robot_1 = int(model.body_rootid[model.geom_bodyid[contact.geom1]]) == robot_root
        robot_2 = int(model.body_rootid[model.geom_bodyid[contact.geom2]]) == robot_root
        if robot_1 == robot_2:
            continue
        other = int(contact.geom2 if robot_1 else contact.geom1)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        external.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                         "normal_force_n": float(force[0]),
                         "distance_m": float(contact.dist)})
    external.sort(key=lambda item: item["normal_force_n"], reverse=True)
    return {"external_contact_count": len(external),
            "external_geom_counts": dict(Counter(item["geom"] for item in external)),
            "strongest_external_contacts": external[:10]}


def tof_evidence(backend: CpuMujocoBamBackend, measurements: dict) -> dict:
    model = backend.model
    data = backend.data
    site_id = backend._tof_site_id
    origin = np.asarray(data.site_xpos[site_id], dtype=float).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    directions = (rotation @ tof_directions().T).T
    root = int(model.body_rootid[backend.policy_inference.trunk_base_id])
    zones = []
    geom_id = np.empty(1, dtype=np.int32)
    for zone, direction in enumerate(directions):
        hit = mujoco.mj_ray(model, data, origin, np.ascontiguousarray(direction),
                            None, 1, -1, geom_id)
        status = measurements["tof_status"][zone]
        distance = measurements["tof_distance_mm"][zone]
        if hit < 0 or hit > 4.0:
            category = "no_target_within_4m"
            geom_name = None
        else:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0]))
            geom_root = int(model.body_rootid[model.geom_bodyid[int(geom_id[0])]])
            category = "self_occluded" if geom_root == root else "external_first_hit"
        if (status == 5) != (category != "no_target_within_4m"):
            raise ValueError("Recorded ToF status differs from the current MuJoCo ray")
        zones.append({"zone": zone, "row": zone // 8, "col": zone % 8,
                      "status": status, "raw_distance_mm": distance,
                      "ray_distance_m": float(hit), "first_hit_geom": geom_name,
                      "category": category})
    counts = Counter(zone["category"] for zone in zones)
    return {"origin_world_m": origin.tolist(),
            "central_forward_world": directions[3 * 8 + 3].tolist(),
            "category_counts": dict(counts),
            "external_first_hit_geoms": dict(Counter(
                zone["first_hit_geom"] for zone in zones if zone["category"] == "external_first_hit")),
            "self_occluded_geoms": dict(Counter(
                zone["first_hit_geom"] for zone in zones if zone["category"] == "self_occluded")),
            "zones": zones}


def snapshot(backend: CpuMujocoBamBackend, measurements: dict, phase: str) -> dict:
    state = {key: measurements[key] for key in
             ("body_position_m", "body_quaternion_wxyz", "body_twist", "body_twist_world",
              "odometry", "height_m", "tilt_rad", "native_contact_count",
              "policy_name", "command_block", "contact_evidence")}
    state.update({"sequence": backend._sequence, "simulation_time_s": float(backend.data.time),
                  "phase": phase, "contacts": contact_evidence(backend),
                  "tof": tof_evidence(backend, measurements)})
    return state


def run(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    backend = CpuMujocoBamBackend(robot_id="collision-diagnosis",
                                  catalog_dir=root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    traces = []
    phases = (("velstand_zero", 187, None),
              ("walk_forward", 538, {"twist": [0.4, 0, 0]}),
              ("walk_stop", 90, {"twist": [0, 0, 0]}),
              ("walk_recovery_side", 265, {"twist": [-0.15, 0.25, 0]}),
              ("walk_recovery_turn", 125, {"twist": [0.1, 0, 0.7]}))
    try:
        first = backend.reset_episode(seed=20260929,
                                      goal={"kind": "room", "room": "office", "hold_ticks": 5})
        traces.append(snapshot(backend, first["measurements"], "initial"))
        for phase, count, command in phases:
            if phase == "walk_forward":
                backend.select_policy("alpha_walking", request_id="diagnosis:select")
            if command is not None:
                backend.set_command(command, request_id=f"diagnosis:command:{phase}")
            for index in range(count):
                inference = backend.infer_policy()
                result = backend.apply_policy_action(
                    inference["action"], request_id=f"diagnosis:{phase}:{index}",
                    expected_sequence=inference["sequence"], should_stop=lambda: False)
                sequence = result["sequence"]
                if sequence in MILESTONES or sequence % 25 == 0:
                    traces.append(snapshot(backend, result["measurements"], phase))
        if backend._sequence != 1205:
            raise ValueError("The diagnostic control schedule did not reach 1205 actions")
    finally:
        backend.close()
    payload = {"source": "official CPU MuJoCo/BAM scene_apartment.xml",
               "seed": 20260929, "spawn_pose": {"x_m": 0, "y_m": 0, "yaw_rad": 0},
               "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
               "phases": [{"name": name, "controls": count, "command": command}
                          for name, count, command in phases],
               "traces": traces}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return {"trace_count": len(traces), "last_sequence": traces[-1]["sequence"],
            "milestones": [{"sequence": trace["sequence"], "phase": trace["phase"],
                            "position_m": trace["body_position_m"],
                            "yaw_rad": trace["odometry"]["yaw_rad"],
                            "contacts": trace["contacts"]["external_geom_counts"],
                            "tof_categories": trace["tof"]["category_counts"]}
                           for trace in traces if trace["sequence"] in MILESTONES]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
