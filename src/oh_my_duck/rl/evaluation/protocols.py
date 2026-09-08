"""Task-specific evaluation batteries; thresholds are separate from training rewards."""

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

    def score(self, trace, completed):
        height, tilt = np.asarray(trace["height"]), np.asarray(trace["tilt"])
        if self.kind == "walking":
            twist = np.asarray(trace["twist"])
            command = np.asarray(trace["command"])
            rmse = np.sqrt(np.mean((twist - command) ** 2, axis=0))
            # Whole-episode RMSE dilutes motion errors with idle time: standing
            # still passes the old 0.1 m/s / 0.5 rad/s limits. Require meaningful
            # signed response in every commanded segment, independently of RMSE.
            boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(command, axis=0), axis=1)) + 1, len(command)]
            responses = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                requested = command[start]
                for axis in np.flatnonzero(np.abs(requested) > 1e-8):
                    measured = float(twist[start:end, axis].mean())
                    responses.append({
                        "start_tick": int(start), "end_tick": int(end),
                        "axis": ("vx", "vy", "yaw_rate")[axis],
                        "command": float(requested[axis]), "measured_mean": measured,
                        "response_fraction": measured / float(requested[axis]),
                    })
            responds = bool(responses) and all(
                row["response_fraction"] >= self.minimum_command_response for row in responses
            )
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
                "scoring_version": 2,
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
