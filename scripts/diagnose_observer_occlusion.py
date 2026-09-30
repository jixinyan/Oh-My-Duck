import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def recorded_poses(export: Path) -> list[dict]:
    events = json.loads((export / "source" / "events.json").read_text())
    poses = []
    for event in events:
        if event["type"] != "tool.completed" or event["detail"].get("tool") != "microduck.read_sensor":
            continue
        result = event["detail"]["result"]
        if result["sensor"] == "odometry":
            poses.append({"sequence": result["sequence"],
                          "odometry": result["measurements"]["odometry"]})
    if len(poses) < 2:
        raise ValueError("Recorded run needs initial and final native odometry")
    return [poses[0], poses[-1]]


def evaluate(backend: CpuMujocoBamBackend, renderer: mujoco.Renderer,
             poses: list[dict]) -> list[dict]:
    results = []
    robot_body = int(backend.model.body_rootid[backend.policy_inference.trunk_base_id])
    for distance in (0.4, 0.5, 0.6):
        for elevation in (-45.0, -55.0, -65.0, -75.0):
            for azimuth in (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0,
                            180.0, 210.0, 225.0, 240.0, 270.0, 300.0, 315.0, 330.0):
                candidate = {"distance_m": distance, "elevation_deg": elevation,
                             "azimuth_deg": azimuth, "poses": []}
                for pose in poses:
                    odometry = pose["odometry"]
                    backend.reset_episode(seed=20260929,
                                          goal={"kind": "room", "room": "office", "hold_ticks": 5},
                                          spawn_pose={"x_m": odometry["x_m"], "y_m": odometry["y_m"],
                                                      "yaw_rad": odometry["yaw_rad"]})
                    target = backend.data.xpos[backend.policy_inference.trunk_base_id].copy()
                    camera = mujoco.MjvCamera()
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = target
                    camera.distance = distance
                    camera.azimuth = azimuth
                    camera.elevation = elevation
                    renderer.update_scene(backend.data, camera=camera)
                    camera_position = np.mean([view.pos for view in renderer.scene.camera], axis=0).astype(float)
                    direction = target - camera_position
                    distance_to_target = float(np.linalg.norm(direction))
                    geom_id = np.empty(1, dtype=np.int32)
                    wall_distance = mujoco.mj_ray(backend.model, backend.data, camera_position,
                                                  direction / distance_to_target, None, 1,
                                                  robot_body, geom_id)
                    clear = wall_distance < 0 or wall_distance >= distance_to_target - 0.01
                    candidate["poses"].append({"sequence": pose["sequence"],
                                               "camera_position_m": camera_position.tolist(),
                                               "target_m": target.tolist(),
                                               "ray_distance_m": float(wall_distance),
                                               "target_distance_m": distance_to_target,
                                               "hit_geom": (mujoco.mj_id2name(backend.model,
                                                                               mujoco.mjtObj.mjOBJ_GEOM,
                                                                               int(geom_id[0]))
                                                            if wall_distance >= 0 else None),
                                               "clear": bool(clear)})
                results.append(candidate)
    return results


def render_shortlist(backend: CpuMujocoBamBackend, renderer: mujoco.Renderer,
                     poses: list[dict], output: Path, candidates: list[dict]) -> list[dict]:
    selected = {(0.6, -55.0, 45.0), (0.6, -55.0, 315.0), (0.6, -55.0, 0.0),
                (0.5, -55.0, 45.0), (0.5, -45.0, 45.0), (0.5, -55.0, 315.0)}
    robot_root = int(backend.model.body_rootid[backend.policy_inference.trunk_base_id])
    robot_geoms = np.flatnonzero(backend.model.body_rootid[backend.model.geom_bodyid] == robot_root)
    rendered = []
    for candidate in candidates:
        key = (candidate["distance_m"], candidate["elevation_deg"], candidate["azimuth_deg"])
        if key not in selected:
            continue
        for pose in poses:
            odometry = pose["odometry"]
            backend.reset_episode(seed=20260929,
                                  goal={"kind": "room", "room": "office", "hold_ticks": 5},
                                  spawn_pose={"x_m": odometry["x_m"], "y_m": odometry["y_m"],
                                              "yaw_rad": odometry["yaw_rad"]})
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = backend.data.xpos[backend.policy_inference.trunk_base_id]
            camera.distance, camera.elevation, camera.azimuth = key
            renderer.update_scene(backend.data, camera=camera)
            rgb = renderer.render().copy()
            image_path = output.parent / (f"observer-d{key[0]:.1f}-e{abs(key[1]):.0f}-a{key[2]:.0f}"
                                          f"-s{pose['sequence']}.png")
            Image.fromarray(rgb).save(image_path)
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            visible = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM.value) &
                       np.isin(segmentation[:, :, 0], robot_geoms))
            visible_geoms = np.unique(segmentation[:, :, 0][visible])
            body_ids = np.unique(backend.model.geom_bodyid[visible_geoms])
            rendered.append({"distance_m": key[0], "elevation_deg": key[1],
                             "azimuth_deg": key[2], "sequence": pose["sequence"],
                             "png": str(image_path), "robot_pixels": int(visible.sum()),
                             "visible_robot_body_names": [mujoco.mj_id2name(
                                 backend.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
                                 for body_id in body_ids]})
    return rendered


def success_position(export: Path) -> list[float]:
    events = json.loads((export / "source" / "events.json").read_text())
    checked = [event for event in events if event["type"] == "verification.checked"]
    fact = next(fact for fact in checked[-1]["detail"]["facts"]
                if fact["check_id"] == "goal_reached")
    if fact["value"] is not True:
        raise ValueError("Additional run must contain a passed native goal")
    return json.loads(fact["reason"])["evidence"]["robot_world_position_m"]


def ray_at_success(backend: CpuMujocoBamBackend, renderer: mujoco.Renderer,
                   position: list[float], candidates: list[dict]) -> list[dict]:
    target = np.asarray(position, dtype=float)
    robot_root = int(backend.model.body_rootid[backend.policy_inference.trunk_base_id])
    checks = []
    for candidate in candidates:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = target
        camera.distance = candidate["distance_m"]
        camera.elevation = candidate["elevation_deg"]
        camera.azimuth = candidate["azimuth_deg"]
        renderer.update_scene(backend.data, camera=camera)
        camera_position = np.mean([view.pos for view in renderer.scene.camera], axis=0).astype(float)
        direction = target - camera_position
        distance = float(np.linalg.norm(direction))
        geom_id = np.empty(1, dtype=np.int32)
        wall_distance = mujoco.mj_ray(backend.model, backend.data, camera_position,
                                      direction / distance, None, 1, robot_root, geom_id)
        checks.append({"distance_m": camera.distance, "elevation_deg": camera.elevation,
                       "azimuth_deg": camera.azimuth,
                       "camera_position_m": camera_position.tolist(),
                       "target_distance_m": distance, "ray_distance_m": float(wall_distance),
                       "hit_geom": (mujoco.mj_id2name(backend.model, mujoco.mjtObj.mjOBJ_GEOM,
                                                       int(geom_id[0])) if wall_distance >= 0 else None),
                       "clear": bool(wall_distance < 0 or wall_distance >= distance - 0.01)})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--success-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    poses = recorded_poses(args.export)
    root = Path(__file__).resolve().parents[1]
    backend = CpuMujocoBamBackend(robot_id="observer-ray-diagnosis",
                                  catalog_dir=root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    try:
        with mujoco.Renderer(backend.model, width=640, height=480) as renderer:
            results = evaluate(backend, renderer, poses)
            rendered = render_shortlist(backend, renderer, poses, args.output, results)
            success = success_position(args.success_export) if args.success_export else None
            success_rays = ray_at_success(backend, renderer, success, results) if success else []
    finally:
        backend.close()
    payload = {"run_id": args.export.name, "recorded_poses": poses,
               "success_run_id": args.success_export.name if args.success_export else None,
               "success_position_m": success, "success_rays": success_rays,
               "candidates": results, "rendered_shortlist": rendered,
               "clear_both": [candidate for candidate in results
                              if all(pose["clear"] for pose in candidate["poses"])]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(results), "clear_both_count": len(payload["clear_both"]),
                      "success_position_m": success,
                      "success_clear_count": sum(ray["clear"] for ray in success_rays),
                      "selected_success_rays": [ray for ray in success_rays if
                                                (ray["distance_m"], ray["elevation_deg"], ray["azimuth_deg"])
                                                in {(0.5, -45.0, 45.0), (0.5, -55.0, 45.0)}]}, indent=2))


if __name__ == "__main__":
    main()
