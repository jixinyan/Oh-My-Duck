"""Simulator-independent command timing for reproducible comparisons."""
from __future__ import annotations

import math

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HOME = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530)


def compile_schedule(config: dict) -> list[tuple[int, str, tuple[float, float, float]]]:
    """Return one command per policy tick, rejecting rounded/ambiguous durations."""
    dt = config["physics_dt"]
    decimation = config["decimation"]
    if not math.isfinite(dt) or dt <= 0 or type(decimation) is not int or decimation < 1:
        raise ValueError("Invalid physics timestep or decimation")
    control_dt = dt * decimation
    if not math.isclose(control_dt, 0.02, abs_tol=1e-10):
        raise ValueError("The official policy contract requires 50 Hz")
    schedule = []
    for segment_id, segment in enumerate(config["segments"]):
        duration = segment["duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Segment durations must be positive and finite")
        ticks = duration / control_dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8):
            raise ValueError("Segment duration must be an integer number of control ticks")
        twist = tuple(segment["twist"])
        if len(twist) != 3 or not all(math.isfinite(x) for x in twist):
            raise ValueError("Twist must contain three finite values in m/s, m/s, rad/s")
        schedule.extend([(segment_id, segment["name"], twist)] * round(ticks))
    if not schedule:
        raise ValueError("At least one command segment is required")
    return schedule
