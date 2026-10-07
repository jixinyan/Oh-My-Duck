from __future__ import annotations

import math
from typing import Any

from oh_my_duck.robotics.microduck.motion_guard import MotionGuard


class MetricMotion:
    DISTANCE_TOLERANCE_M = 0.05
    ANGLE_TOLERANCE_DEG = 5.0

    def __init__(self, operation: str, arguments: dict[str, Any], state: dict[str, Any],
                 robot_model: str = "allcollisions"):
        self.operation = operation
        self.start_position = list(state["body_position_m"])
        self.start_yaw = float(state["odometry"]["yaw_rad"])
        self.last_yaw = self.start_yaw
        self.unwrapped_yaw = self.start_yaw
        self.last_sequence = -1
        self.phase = "moving"
        self.result: dict[str, Any] | None = None
        self.request_id: str | None = None
        self.correction_attempts = 0
        self.rotation_settling_delta = 0.0
        self.rotation_braking_yaw = None
        self.progress_history: list[tuple[int, float]] = []
        self.robot_model = robot_model
        if operation == "walk":
            self.amount = self._number(arguments, "distance_m", -10, 10)
            if abs(self.amount) < 0.1:
                raise ValueError("Walking distance must have magnitude at least 0.1 m")
            self.speed = self._number(arguments, "speed_m_s", 0.1, 0.4, 0.4)
            self.target = [self.start_position[0] + self.amount * math.cos(self.start_yaw),
                           self.start_position[1] + self.amount * math.sin(self.start_yaw)]
            self.previous_distance_error = self.amount
        elif operation == "rotate":
            self.amount = self._number(arguments, "angle_deg", -360, 360)
            if abs(self.amount) < 10:
                raise ValueError("Rotation angle must have magnitude at least 10 degrees")
            self.speed = math.radians(self._number(arguments, "angular_speed_deg_s", 10, 55, 45))
            self.target = self.start_yaw + math.radians(self.amount)
            self.previous_angle_error = math.radians(self.amount)
        else:
            raise ValueError("Unknown metric motion operation")

    @staticmethod
    def _number(arguments, key, minimum, maximum, default=None):
        value = arguments.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be finite and between {minimum} and {maximum}")
        return float(value)

    def command(self, state):
        if self.phase not in {"moving", "braking"}:
            return {"twist": [0.0, 0.0, 0.0]}
        if self.operation == "walk":
            position = state["body_position_m"]
            remaining = ((self.target[0] - position[0]) * math.cos(self.start_yaw) +
                         (self.target[1] - position[1]) * math.sin(self.start_yaw))
            world_velocity = state["body_twist_world"]
            velocity = world_velocity[0] * math.cos(self.start_yaw) + world_velocity[1] * math.sin(self.start_yaw)
            cross_track = (-(self.target[0] - position[0]) * math.sin(self.start_yaw) +
                           (self.target[1] - position[1]) * math.cos(self.start_yaw))
            if self.phase == "braking":
                forward = 0.0
            elif self.robot_model == "groundcontact_rollers":
                forward = max(-self.speed, min(self.speed, 2.0 * remaining - 1.5 * velocity))
                if abs(forward) < 0.3:
                    forward = math.copysign(min(self.speed, 0.3), remaining)
            else:
                forward = math.copysign(self.speed, remaining)
            heading = self.start_yaw
            lateral = 0.0
            if self.robot_model == "allcollisions" and self.phase == "moving":
                heading += math.atan2(math.copysign(1.0, remaining) * cross_track,
                                      max(0.3, abs(remaining)))
                lateral = max(-0.3, min(0.3, 2.0 * cross_track))
            yaw_error = math.atan2(math.sin(heading - self.last_yaw),
                                   math.cos(heading - self.last_yaw))
            return {"twist": [round(forward, 2), round(lateral, 2),
                               0.0 if self.phase == "braking" else round(max(-0.6, min(0.6, 1.5 * yaw_error)), 2)]}
        if self.phase == "braking":
            return {"twist": [0.0, 0.0, 0.0]}
        remaining = self.target - self.rotation_settling_delta - self.unwrapped_yaw
        if self.robot_model == "groundcontact_rollers":
            return {"twist": [0.0, 0.0, round(math.copysign(self.speed, remaining), 2)]}
        return {"twist": [0.2, 0.25, round(math.copysign(self.speed, remaining), 2)]}

    def observe_progress(self, sequence, error):
        self.progress_history.append((sequence, error))
        while (self.progress_history and
               sequence - self.progress_history[0][0] > MotionGuard.STALL_WINDOW_STEPS):
            self.progress_history.pop(0)
        if sequence - self.progress_history[0][0] < MotionGuard.STALL_WINDOW_STEPS:
            return False
        threshold = (MotionGuard.STALL_TRANSLATION_M if self.operation == "walk" else
                     math.degrees(MotionGuard.STALL_YAW_RAD))
        return self.progress_history[0][1] - error < threshold

    def observe(self, sample, state, stopped_samples):
        sequence = sample["sequence"]
        if sequence <= self.last_sequence:
            raise RuntimeError("Metric motion requires a new physical control sample")
        yaw = sample["yaw_rad"]
        self.unwrapped_yaw += math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
        self.last_yaw, self.last_sequence = yaw, sequence
        if self.operation == "walk":
            position = sample["body_position_m"]
            error = math.hypot(position[0] - self.target[0], position[1] - self.target[1])
            progress = ((position[0] - self.start_position[0]) * math.cos(self.start_yaw) +
                        (position[1] - self.start_position[1]) * math.sin(self.start_yaw))
            remaining = self.amount - progress
            reached = error <= 0.025 or (self.previous_distance_error * remaining <= 0 and
                                        (self.robot_model != "allcollisions" or
                                         error <= self.DISTANCE_TOLERANCE_M))
            if self.robot_model == "allcollisions":
                velocity = state["body_twist_world"]
                along_speed = velocity[0] * math.cos(self.start_yaw) + velocity[1] * math.sin(self.start_yaw)
                cross_track = math.sqrt(max(0.0, error ** 2 - remaining ** 2))
                reached = reached or (cross_track <= 0.025 and
                                     abs(remaining) <= max(0.025, 0.18 * abs(along_speed)))
            elif self.robot_model == "groundcontact_rollers":
                velocity = state["body_twist_world"]
                along_speed = velocity[0] * math.cos(self.start_yaw) + velocity[1] * math.sin(self.start_yaw)
                reached = reached or (remaining * along_speed > 0 and
                                     abs(remaining) <= max(0.025, 1.3 * abs(along_speed)))
            self.previous_distance_error = remaining
            tolerance = self.DISTANCE_TOLERANCE_M
            progress_error = error
        else:
            angle_error = self.target - self.unwrapped_yaw
            error = abs(math.degrees(angle_error))
            progress = math.degrees(self.unwrapped_yaw - self.start_yaw)
            aim_error = angle_error - self.rotation_settling_delta
            reached = abs(math.degrees(aim_error)) <= 0.5 or self.previous_angle_error * aim_error <= 0
            self.previous_angle_error = aim_error
            tolerance = self.ANGLE_TOLERANCE_DEG
            progress_error = abs(math.degrees(aim_error))
        self.result = {"operation": self.operation, "requested": self.amount,
                       "request_id": self.request_id,
                       "unit": "m" if self.operation == "walk" else "deg",
                       "measured": progress, "error": error, "tolerance": tolerance,
                       "start_position_m": self.start_position, "start_yaw_rad": self.start_yaw,
                       "body_position_m": list(sample["body_position_m"]),
                       "translation_xy_m": math.hypot(sample["body_position_m"][0] - self.start_position[0],
                                                       sample["body_position_m"][1] - self.start_position[1]),
                       "yaw_rad": yaw, "unwrapped_yaw_rad": self.unwrapped_yaw,
                       "sequence": sequence, "stopped_samples": stopped_samples,
                       "phase": self.phase, "completed": False}
        if state["fallen"]:
            self.phase = "failed"
        # 目标范围内已经停止时执行零命令制动，随后重新测量最终误差。
        elif self.phase == "moving" and (reached or (error <= tolerance and stopped_samples >= 5)):
            self.phase = "braking"
            self.progress_history.clear()
            if self.operation == "rotate":
                self.rotation_braking_yaw = self.unwrapped_yaw
        elif self.phase == "moving" and self.observe_progress(sequence, progress_error):
            self.phase = "failed"
            self.result["reason"] = "metric_progress_stalled"
        elif self.phase == "braking" and stopped_samples >= 5:
            if error <= tolerance:
                self.phase = "complete"
            elif self.correction_attempts < 3:
                self.correction_attempts += 1
                if self.operation == "rotate":
                    self.rotation_settling_delta = self.unwrapped_yaw - self.rotation_braking_yaw
                    self.previous_angle_error = self.target - self.rotation_settling_delta - self.unwrapped_yaw
                self.phase = "moving"
                self.progress_history.clear()
            else:
                self.phase = "failed"
        self.result.update(phase=self.phase, completed=self.phase == "complete",
                           correction_attempts=self.correction_attempts)
        if self.operation == "rotate":
            self.result["measured_settling_delta_deg"] = math.degrees(self.rotation_settling_delta)
        return self.result
