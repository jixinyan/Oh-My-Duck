from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping


class MotionGuard:
    STALL_WINDOW_STEPS = 50
    STALL_TRANSLATION_M = 0.01
    STALL_YAW_RAD = 0.05
    FORWARD_TOF_STOP_MM = 90

    def __init__(self, command: Mapping[str, Any], effective_after_sequence: int,
                 max_control_steps: int) -> None:
        if type(effective_after_sequence) is not int or effective_after_sequence < 0:
            raise ValueError("Motion guard requires a native control sequence")
        if type(max_control_steps) is not int or not 5 <= max_control_steps <= 100:
            raise ValueError("Motion guard requires 5 to 100 control steps")
        self.command = dict(command)
        self.effective_after_sequence = effective_after_sequence
        self.max_control_steps = max_control_steps
        self._history: list[tuple[int, float, float, float]] = []

    def update_command(self, command: Mapping[str, Any], effective_after_sequence: int,
                       max_control_steps: int) -> None:
        updated = MotionGuard(command, effective_after_sequence, max_control_steps)
        if effective_after_sequence < self.effective_after_sequence:
            raise ValueError("Motion command sequence moved backwards")
        self.command = updated.command
        self.effective_after_sequence = updated.effective_after_sequence
        self.max_control_steps = updated.max_control_steps

    def observe(self, sample: Mapping[str, Any]) -> dict[str, Any] | None:
        sequence = sample["sequence"]
        position = sample["body_position_m"]
        yaw = sample["yaw_rad"]
        distances = sample["tof_distance_mm"]
        statuses = sample["tof_status"]
        if (type(sequence) is not int or len(position) != 3 or
                any(not math.isfinite(value) for value in position) or
                not math.isfinite(yaw) or len(distances) != 64 or len(statuses) != 64 or
                any(type(distance) is not int or not 0 <= distance <= 4000
                    for distance in distances) or
                any(type(status) is not int for status in statuses)):
            raise ValueError("Motion guard received an invalid physical observation")
        used = sequence - self.effective_after_sequence
        if used <= 0:
            return None
        if used > self.max_control_steps:
            raise RuntimeError("Motion command exceeded its admitted control-step limit")
        self._history.append((sequence, position[0], position[1], yaw))
        if len(self._history) > self.STALL_WINDOW_STEPS + 1:
            self._history.pop(0)
        central = [distances[row * 8 + col]
                   for row in range(2, 6) for col in range(2, 6)
                   if statuses[row * 8 + col] == 5]
        closest = min(central) if central else None
        central_median = median(central) if central else None
        unknown_statuses = sorted(set(statuses) - {5, 255})
        twist = self.command["twist"]
        contact = sample["contact_evidence"]
        external = contact["current_control_non_ground_external"]
        reason = None
        if external:
            reason = "external_contact"
        elif twist[0] > 0 and unknown_statuses:
            reason = "tof_invalid"
        elif twist[0] > 0 and closest is not None and closest < self.FORWARD_TOF_STOP_MM:
            reason = "forward_proximity"
        elif (any(abs(value) > 1e-6 for value in twist) and
              len(self._history) > self.STALL_WINDOW_STEPS and
              self._history[-1][0] - self._history[0][0] >= self.STALL_WINDOW_STEPS):
            translation = math.hypot(self._history[-1][1] - self._history[0][1],
                                     self._history[-1][2] - self._history[0][2])
            turn = math.atan2(math.sin(self._history[-1][3] - self._history[0][3]),
                              math.cos(self._history[-1][3] - self._history[0][3]))
            if translation < self.STALL_TRANSLATION_M and abs(turn) < self.STALL_YAW_RAD:
                reason = "motion_stalled"
        if reason is None and used >= self.max_control_steps:
            reason = "command_segment_complete"
        if reason is None:
            return None
        return {
            "reason": reason, "episode_id": sample["episode_id"], "sequence": sequence,
            "command": self.command, "used_control_steps": used,
            "max_control_steps": self.max_control_steps,
            "body_position_m": list(position), "yaw_rad": yaw,
            "central_tof_valid_count": len(central),
            "tof_unknown_statuses": unknown_statuses,
            "central_tof_closest_mm": closest,
            "central_tof_median_mm": central_median,
            "forward_tof_stop_mm": self.FORWARD_TOF_STOP_MM,
            "contact_evidence": contact,
            "stall_window_steps": self.STALL_WINDOW_STEPS,
            "stall_translation_m": self.STALL_TRANSLATION_M,
            "stall_yaw_rad": self.STALL_YAW_RAD,
        }
