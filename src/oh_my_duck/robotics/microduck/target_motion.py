from copy import deepcopy
from datetime import datetime, timezone
import math

from oh_my_duck.robotics.microduck.metric_motion import MetricMotion


class TargetApproachMotion:
    DISTANCE_TOLERANCE_M = 0.05
    MAX_OBSERVATION_AGE_MS = 60000

    def __init__(self, arguments: dict, observation: dict, state: dict, *,
                 episode_id: str, sequence: int, robot_model: str):
        if set(arguments) - {"target_id", "standoff_m", "speed_m_s", "max_age_ms"}:
            raise ValueError("目标接近参数包含未知字段")
        target_id = arguments["target_id"]
        if not isinstance(target_id, str) or not target_id or len(target_id) > 256:
            raise ValueError("target_id 必须包含 1–256 个字符")
        if robot_model not in {"allcollisions", "groundcontact_rollers"}:
            raise ValueError("目标接近要求已声明的机器人模型")
        if observation["episode_id"] != episode_id or observation["sequence"] != sequence:
            raise ValueError("目标观测必须属于当前 episode 和物理 sequence")
        observed = datetime.fromisoformat(observation["observed_at"])
        if observed.utcoffset() is None:
            raise ValueError("目标观测时间必须具有 UTC 时区")
        age_ms = (datetime.now(timezone.utc) - observed).total_seconds() * 1000
        max_age_ms = arguments.get("max_age_ms", self.MAX_OBSERVATION_AGE_MS)
        if type(max_age_ms) is not int or not 100 <= max_age_ms <= self.MAX_OBSERVATION_AGE_MS:
            raise ValueError("max_age_ms 必须为 100–60000 的整数")
        if not 0 <= age_ms <= max_age_ms:
            raise ValueError("目标观测已经超过允许的时间")
        targets = observation["targets"]
        identifiers = [target["target_id"] for target in targets]
        if len(set(identifiers)) != len(identifiers) or target_id not in identifiers:
            raise ValueError("目标必须出现在当前观测的唯一 target_id 清单中")
        target = targets[identifiers.index(target_id)]
        if target["distance_status"] != "valid" or target["valid_depth_pixels"] < 8:
            raise ValueError("目标接近要求有效的 RGBD 测量")
        point = target["surface_position_world_m"]
        if len(point) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) for value in point):
            raise ValueError("目标表面位置必须包含三个有限的 world 坐标")
        position = state["body_position_m"]
        distance = math.hypot(point[0] - position[0], point[1] - position[1])
        if not math.isclose(distance, target["distance_xy_m"], rel_tol=0, abs_tol=1e-4):
            raise ValueError("目标距离与当前机器人坐标不一致")
        self.amount = MetricMotion._number(arguments, "standoff_m", 0.2, 2.0)
        self.speed = MetricMotion._number(arguments, "speed_m_s", 0.1, 0.4, 0.4)
        if distance > 10:
            raise ValueError("目标接近要求十米以内的实际观测")
        self.operation = "approach_object"
        self.start_position = list(position)
        self.start_yaw = state["odometry"]["yaw_rad"]
        self.target_point = list(point)
        self.target_observation = deepcopy({key: observation[key] for key in
            ("episode_id", "sequence", "observed_at", "prompt", "source")})
        self.target_observation["target"] = deepcopy(target)
        self.target_observation.update(age_ms_at_admission=age_ms, max_age_ms=max_age_ms)
        self.robot_model = robot_model
        self.phase = "moving"
        self.request_id = None
        self.last_sequence = sequence
        self.correction_attempts = 0
        self.result = None

    def geometry(self, state):
        position = state["body_position_m"]
        yaw = state["odometry"]["yaw_rad"]
        distance = math.hypot(self.target_point[0] - position[0], self.target_point[1] - position[1])
        bearing = math.atan2(self.target_point[1] - position[1], self.target_point[0] - position[0]) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return distance, bearing

    def command(self, state):
        if self.phase != "moving":
            return {"twist": [0.0, 0.0, 0.0]}
        distance, bearing = self.geometry(state)
        error = distance - self.amount
        if abs(error) <= 0.025:
            return {"twist": [0.0, 0.0, 0.0]}
        yaw = max(-0.6, min(0.6, 1.5 * bearing))
        if abs(bearing) > math.radians(20):
            if self.robot_model == "groundcontact_rollers":
                return {"twist": [0.0, 0.0, round(yaw, 2)]}
            return {"twist": [0.2, 0.25, round(yaw, 2)]}
        return {"twist": [round(math.copysign(self.speed, error), 2), 0.0, round(yaw, 2)]}

    def observe(self, sample, state, stopped_samples):
        if sample["sequence"] <= self.last_sequence:
            raise RuntimeError("目标接近要求新的实际控制步骤")
        self.last_sequence = sample["sequence"]
        distance, bearing = self.geometry(state)
        error = abs(distance - self.amount)
        velocity = state["body_twist_world"]
        position = state["body_position_m"]
        radial_velocity = (0.0 if distance == 0 else
            (velocity[0] * (self.target_point[0] - position[0]) +
             velocity[1] * (self.target_point[1] - position[1])) / distance)
        if state["fallen"]:
            self.phase = "failed"
        elif self.phase == "moving" and (
                error <= 0.025 or (error <= self.DISTANCE_TOLERANCE_M and stopped_samples >= 5) or
                (radial_velocity * (distance - self.amount) > 0 and
                 error <= min(self.DISTANCE_TOLERANCE_M, 0.18 * abs(radial_velocity)))):
            self.phase = "braking"
        elif self.phase == "braking" and stopped_samples >= 5:
            if error <= self.DISTANCE_TOLERANCE_M:
                self.phase = "complete"
            elif self.correction_attempts < 3:
                self.phase = "moving"
                self.correction_attempts += 1
            else:
                self.phase = "failed"
        self.result = {
            "operation": self.operation, "request_id": self.request_id,
            "target_observation": self.target_observation,
            "requested": self.amount, "unit": "m", "measured": distance,
            "error": error, "tolerance": self.DISTANCE_TOLERANCE_M,
            "bearing_deg": math.degrees(bearing), "surface_position_world_m": self.target_point,
            "start_position_m": self.start_position, "start_yaw_rad": self.start_yaw,
            "body_position_m": list(position), "sequence": sample["sequence"],
            "translation_xy_m": math.hypot(position[0] - self.start_position[0],
                                            position[1] - self.start_position[1]),
            "stopped_samples": stopped_samples, "correction_attempts": self.correction_attempts,
            "phase": self.phase, "completed": self.phase == "complete",
        }
        return self.result
