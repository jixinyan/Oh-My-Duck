import math

import numpy as np


def measure_target(mask, points_world_m, camera_position_m, body_position_m, yaw_rad):
    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(points_world_m, dtype=np.float32)
    if points.shape != (*mask.shape, 3) or mask.ndim != 2:
        raise ValueError("Mask and calibrated world points must share the same image geometry")
    valid = mask & np.isfinite(points).all(axis=-1)
    selected = points[valid]
    if len(selected) < 8:
        return {"distance_status": "insufficient_valid_depth", "valid_depth_pixels": len(selected)}
    center = np.median(selected, axis=0)
    camera = np.asarray(camera_position_m, dtype=float)
    body = np.asarray(body_position_m, dtype=float)
    if camera.shape != (3,) or body.shape != (3,) or not np.isfinite(camera).all() or not np.isfinite(body).all():
        raise ValueError("Camera and body positions must contain three finite meters")
    bearing = math.atan2(float(center[1] - body[1]), float(center[0] - body[0])) - yaw_rad
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    distances = np.linalg.norm(selected - camera, axis=-1)
    return {"distance_status": "valid", "valid_depth_pixels": len(selected),
            "distance_m": float(np.median(distances)),
            "distance_interval_m": np.quantile(distances, [0.1, 0.9]).tolist(),
            "distance_xy_m": float(np.linalg.norm(center[:2] - body[:2])),
            "bearing_deg": math.degrees(bearing), "surface_position_world_m": center.tolist()}
