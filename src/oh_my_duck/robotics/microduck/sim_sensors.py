# Copyright 2026 Pollen Robotics. Licensed under Apache-2.0.
# Modified by Oh My Duck: owned, framework-neutral sensor adapters.
"""Deterministic simulated Microduck camera and ToF sensor primitives.

These helpers mirror the current official simulator sensor conventions without
starting a daemon or changing the RL task loop. They are intentionally small so
evaluation, a future external harness, and a hardware protocol adapter can use
the same byte/ray conventions.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

CAMERA_Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)
CAMERA_U = np.array([-0.168736, -0.331264, 0.5], dtype=np.float32)
CAMERA_V = np.array([0.5, -0.418688, -0.081312], dtype=np.float32)

TOF_ROWS = 8
TOF_COLS = 8
TOF_FOV_DEG = 45.0
TOF_MAX_RANGE_M = 4.0
TOF_STATUS_VALID = 5
TOF_STATUS_NO_TARGET = 255


def camera_optical_pose(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    site = model.site("head_camera").id
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.site_quat[site])
    # site 的 +X 前向与 +Z 上向转换为 OpenGL 相机方向。
    optical = rotation.reshape(3, 3) @ np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, optical.ravel())
    return model.site_pos[site].copy(), quaternion


def to_uyvy(rgb: np.ndarray) -> bytes:
    """Pack an even-width RGB frame as UYVY 4:2:2 bytes.

    Chroma is averaged across each pair, matching the camera pipeline used by
    the official simulator and avoiding a one-pixel colour phase shift.
    """
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape [height, width, 3], got {frame.shape}")
    if frame.shape[1] % 2:
        raise ValueError("UYVY requires an even image width")
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError("RGB frame must contain numeric values")
    frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    luma = frame @ CAMERA_Y
    chroma_u = frame @ CAMERA_U + 128.0
    chroma_v = frame @ CAMERA_V + 128.0
    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) * 0.5, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) * 0.5, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


def tof_directions(
    rows: int = TOF_ROWS, cols: int = TOF_COLS, fov_deg: float = TOF_FOV_DEG
) -> np.ndarray:
    """Return zone-centre unit rays in the ToF site's ``+x,+y,+z`` frame.

    Row zero is the top of the image and column zero is its left side, matching
    the real VL53L5CX frame consumed by the mapping stack.
    """
    if rows <= 0 or cols <= 0 or not math.isfinite(fov_deg) or not 0.0 < fov_deg < 180.0:
        raise ValueError("rows/cols must be positive and fov_deg must be in (0, 180)")
    half = math.radians(fov_deg) * 0.5
    row_edges = np.linspace(-half, half, rows + 1)
    col_edges = np.linspace(-half, half, cols + 1)
    elevations = (row_edges[:-1] + row_edges[1:]) * 0.5
    azimuths = (col_edges[:-1] + col_edges[1:]) * 0.5
    rays = np.empty((rows * cols, 3), dtype=np.float64)
    for row, centre_elevation in enumerate(elevations):
        elevation = -centre_elevation
        for col, centre_azimuth in enumerate(azimuths):
            # Sensor buffers are top-to-bottom and left-to-right; the physical
            # site uses +z up and +y left, hence both axes are negated.
            azimuth = -centre_azimuth
            rays[row * cols + col] = (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            )
    return rays


def tof_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    *,
    directions: np.ndarray | None = None,
    max_range_m: float = TOF_MAX_RANGE_M,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[int]]:
    """Cast a simulated 8×8 ToF frame with per-ray finite/zero guards.

    MuJoCo aborts the process when ``mj_ray`` receives a zero-length direction.
    A site can have an uninitialised orientation before the first forward pass,
    so each ray is checked independently; one bad zone does not discard all
    valid zones or take down the evaluator.
    """
    if not math.isfinite(max_range_m) or max_range_m <= 0.0:
        raise ValueError("max_range_m must be finite and positive")
    directions = tof_directions() if directions is None else np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape [zones, 3], got {directions.shape}")
    origin = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    world = (rotation @ directions.T).T
    distances = [0] * len(world)
    statuses = [TOF_STATUS_NO_TARGET] * len(world)
    if not np.isfinite(origin).all():
        return distances, statuses
    random = np.random.default_rng() if rng is None else rng
    geom = np.zeros(1, dtype=np.int32)
    for zone, ray in enumerate(world):
        norm = float(np.linalg.norm(ray))
        if not math.isfinite(norm) or norm < 1e-9:
            continue
        hit = mujoco.mj_ray(model, data, origin, np.ascontiguousarray(ray), None, 1, -1, geom)
        if hit < 0.0 or hit > max_range_m:
            continue
        sigma = 0.003 + 0.02 * (hit / max_range_m)
        measured = min(max_range_m, max(0.0, float(hit) + float(random.normal(0.0, sigma))))
        distances[zone] = int(measured * 1000.0)
        statuses[zone] = TOF_STATUS_VALID
    return distances, statuses
