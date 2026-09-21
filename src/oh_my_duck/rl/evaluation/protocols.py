from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    commands: tuple[tuple[float, float, float], ...]
    reset_probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class EvaluationProtocol:
    scenarios: tuple[Scenario, ...]
    kind: str
    minimum_height: float
    maximum_tilt: float
    hold_seconds: float = 1.0
    minimum_command_response: float = 0.5
    maximum_command_response: float = 1.5
    settling_seconds: float = 0.5
    window_seconds: float = 0.5
    zero_command_limits: tuple[float, float, float] = (0.02, 0.02, 0.1)

    def score(self, trace, completed):
        height, tilt = np.asarray(trace["height"]), np.asarray(trace["tilt"])
        if self.kind == "walking":
            twist = np.asarray(trace["twist"])
            command = np.asarray(trace["command"])
            if (height.ndim != 1 or len(height) == 0 or tilt.shape != height.shape
                    or twist.shape != (len(height), 3) or command.shape != twist.shape):
                raise ValueError("Walking trace requires matching nonempty height, tilt and three-axis samples")
            if not all(np.isfinite(values).all() for values in (height, tilt, twist, command)):
                raise ValueError("Walking trace contains non-finite values")
            rmse = np.sqrt(np.mean((twist - command) ** 2, axis=0))
            boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(command, axis=0), axis=1)) + 1, len(command)]
            responses = []
            stages = []
            settling_ticks = round(self.settling_seconds / 0.02)
            window_ticks = round(self.window_seconds / 0.02)
            if settling_ticks < 0 or window_ticks < 1:
                raise ValueError("Invalid Walking settling or scoring window duration")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                requested = command[start]
                scoring_start = min(start + settling_ticks, end)
                samples = twist[scoring_start:end]
                moving = np.abs(requested) > 1e-8
                sufficient = len(samples) >= window_ticks
                stage = {
                    "start_tick": int(start), "end_tick": int(end),
                    "scoring_start_tick": int(scoring_start), "command": requested.tolist(),
                    "kind": "motion" if moving.any() else "stop",
                    "success": False, "window_count": 0,
                }
                if sufficient:
                    # 每个连续窗口都参与判定，包含阶段末尾的样本。
                    windows = np.lib.stride_tricks.sliding_window_view(samples, window_ticks, axis=0)
                    means = windows.mean(axis=-1)
                    limits = np.asarray(self.zero_command_limits)
                    if moving.any():
                        ratios = means[:, moving] / requested[moving]
                        tracked = ((ratios >= self.minimum_command_response - 1e-12)
                                   & (ratios <= self.maximum_command_response + 1e-12)).all()
                        quiet = (np.abs(means[:, ~moving]) <= limits[~moving] + 1e-12).all()
                        stage["window_mean_min"] = means.min(axis=0).tolist()
                        stage["window_mean_max"] = means.max(axis=0).tolist()
                    else:
                        rms = np.sqrt(np.mean(windows ** 2, axis=-1))
                        tracked = True
                        quiet = (rms <= limits + 1e-12).all()
                        stage["window_rms_max"] = rms.max(axis=0).tolist()
                    stage.update(success=bool(tracked and quiet), window_count=len(windows))
                stages.append(stage)
                for axis in np.flatnonzero(np.abs(requested) > 1e-8):
                    measured = float(samples[:, axis].mean()) if sufficient else None
                    responses.append({
                        "start_tick": int(start), "end_tick": int(end),
                        "axis": ("vx", "vy", "yaw_rate")[axis],
                        "command": float(requested[axis]), "measured_mean": measured,
                        "response_fraction": measured / float(requested[axis]) if sufficient else None,
                    })
            responds = bool(responses) and all(stage["success"] for stage in stages)
            return {
                "success": bool(
                    completed
                    and (height >= self.minimum_height).all()
                    and (tilt <= self.maximum_tilt).all()
                    and (rmse <= [0.1, 0.1, 0.5]).all()
                    and responds
                ),
                "twist_rmse": rmse.tolist(),
                "command_response": responses,
                "minimum_command_response": self.minimum_command_response,
                "maximum_command_response": self.maximum_command_response,
                "settling_seconds": self.settling_seconds,
                "window_seconds": self.window_seconds,
                "zero_command_limits": list(self.zero_command_limits),
                "stages": stages,
                "scoring_version": 3,
                "min_height_m": float(height.min()),
                "max_tilt_rad": float(tilt.max()),
            }
        count = round(self.hold_seconds / 0.02)
        success = (
            len(height) >= count
            and (height[-count:] >= self.minimum_height).all()
            and (tilt[-count:] <= self.maximum_tilt).all()
        )
        return {
            "success": bool(completed and success),
            "final_height_m": float(height[-1]),
            "final_tilt_rad": float(tilt[-1]),
            "hold_seconds_required": self.hold_seconds,
        }


def walking():
    segments = (
        (2.0, (0.0, 0.0, 0.0)),
        (4.0, (0.1, 0.0, 0.0)),
        (2.0, (0.0, 0.0, 0.0)),
        (4.0, (0.0, 0.0, 0.5)),
        (2.0, (0.0, 0.0, 0.0)),
    )
    commands = tuple(command for seconds, command in segments for _ in range(round(seconds / 0.02)))
    return EvaluationProtocol((Scenario("hold_forward_stop_turn", commands),), "walking", 0.065, np.pi / 3)


def standup():
    poses = ("standing", "sitting", "face_down", "face_up")
    return EvaluationProtocol(
        tuple(
            Scenario(pose, ((0.0, 0.0, 0.0),) * 400, {name + "_prob": float(name == pose) for name in poses})
            for pose in poses
        ),
        "standup",
        0.10,
        0.35,
    )
