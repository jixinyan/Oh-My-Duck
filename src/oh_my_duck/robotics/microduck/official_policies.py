from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from oh_my_duck.robotics.policies.joint_onnx import load_joint_session
from oh_my_duck.robotics.microduck.policy_sources import (
    MANIFEST_SHA256, OFFICIAL_REPO, OFFICIAL_REVISION, POLICY_SHA256,
)

APARTMENT_BEHAVIOR_STATUS = {
    "alpha_walking": "command_tracking_failed_vx_0_1_to_0_4_m_s",
    "velstand": "command_tracking_failed_vx_0_1_to_0_4_m_s",
    "alpha_stand": "standing_hold_and_pose_response_observed_recovery_unverified",
    "roller": "requires_roller_robot_model",
    "sitstand": "sit_stand_height_transition_observed",
    "ground_pick": "duration_executed_object_pick_unverified",
    "crouch": "requires_roller_robot_model",
    "roulade": "duration_executed_full_roll_unverified",
    "kick_left": "duration_executed_ball_contact_unverified",
    "kick_right": "duration_executed_ball_contact_unverified",
}

POLICY_COMMAND_CHANNELS = {
    "alpha_walking": ("twist", "head", "body"),
    "velstand": ("twist", "head", "body"),
    "alpha_stand": ("head", "body"),
    "roller": ("twist",),
    "sitstand": ("posture", "head"),
    "ground_pick": ("phase",),
    "crouch": ("phase",),
    "roulade": (),
    "kick_left": (),
    "kick_right": (),
}


@dataclass(frozen=True)
class OfficialPolicy:
    name: str
    file: str
    sha256: str
    kind: str
    encoding: str
    action_scale: float
    duration_s: float | None
    mode: str
    chain: bool
    slot: str | None
    entry_pose: str | None
    ramp_s: float | None
    unwind_s: float | None
    command: dict
    session: object
    command_channels: tuple[str, ...] | None = None


class OfficialPolicyCatalogue:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        manifest_path = self.directory / "manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Official policy manifest SHA-256 differs from the pinned revision")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version"), manifest.get("model_api"), manifest.get("obs_len"),
                manifest.get("action_len")) != (2, 1, 61, 14):
            raise ValueError("Official policy manifest API or dimensions differ")
        if manifest.get("robot") != {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}:
            raise ValueError("Official policy robot metadata differs")
        entries = manifest.get("policies")
        if not isinstance(entries, list) or len(entries) != len(POLICY_SHA256):
            raise ValueError("Official policy count differs")
        self.policies: dict[str, OfficialPolicy] = {}
        files = set()
        for item in entries:
            file = item["file"]
            if file not in POLICY_SHA256 or file in files or Path(file).name != file:
                raise ValueError(f"Unexpected or duplicate official policy file: {file}")
            files.add(file)
            path = self.directory / file
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != POLICY_SHA256[file]:
                raise ValueError(f"Official policy SHA-256 differs: {file}")
            session = load_joint_session(path)
            name = item.get("name", Path(file).stem)
            if name in self.policies:
                raise ValueError(f"Duplicate official policy name: {name}")
            command = item.get("command", {})
            encoding = command.get("encoding", "constant")
            if encoding not in {"constant", "phase", "posture_flag"}:
                raise ValueError(f"Unsupported command encoding: {encoding}")
            kind = item["kind"]
            if kind not in {"perpetual", "episodic", "scripted"}:
                raise ValueError(f"Unsupported official policy kind: {kind}")
            scale = float(item.get("action_scale", 1.0))
            if not math.isfinite(scale) or scale <= 0 or scale > 2.0:
                raise ValueError(f"Invalid action scale: {file}")
            duration = item.get("duration_s")
            if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Invalid duration: {file}")
            for key in ("ramp_s", "unwind_s"):
                timing = item.get(key)
                if timing is not None and (isinstance(timing, bool) or not isinstance(timing, (int, float))
                                           or not math.isfinite(timing) or timing <= 0):
                    raise ValueError(f"Invalid {key}: {file}")
            if "chain" in item and not isinstance(item["chain"], bool):
                raise ValueError(f"Invalid chain: {file}")
            for key in ("slot", "entry_pose"):
                if key in item and (not isinstance(item[key], str) or not item[key]):
                    raise ValueError(f"Invalid {key}: {file}")
            self.policies[name] = OfficialPolicy(name, file, digest, kind, encoding, scale,
                                                 float(duration) if duration is not None else None,
                                                 item.get("mode", "walk"), bool(item.get("chain", False)),
                                                 item.get("slot"), item.get("entry_pose"),
                                                 item.get("ramp_s"), item.get("unwind_s"),
                                                 command, session)
        if files != set(POLICY_SHA256):
            raise ValueError("Official policy set is incomplete")
        if set(self.policies) != set(APARTMENT_BEHAVIOR_STATUS):
            raise ValueError("Official apartment behavior inventory is incomplete")

    def get(self, name: str) -> OfficialPolicy:
        if name not in self.policies:
            raise ValueError(f"Unknown official policy: {name}")
        return self.policies[name]

    def describe_apartment(self) -> list[dict]:
        return [
            {"name": policy.name, "file": policy.file, "sha256": policy.sha256,
             "kind": policy.kind, "encoding": policy.encoding,
             "action_scale": policy.action_scale, "duration_s": policy.duration_s,
             "ramp_s": policy.ramp_s, "unwind_s": policy.unwind_s,
             "entry_pose": policy.entry_pose, "slot": policy.slot, "chain": policy.chain,
             "command": policy.command, "command_channels": list(POLICY_COMMAND_CHANNELS[policy.name]),
             "required_robot_mode": "groundcontact_rollers" if policy.mode == "roller"
                                    else "standard_14_joint_microduck",
             "executable_here": policy.mode != "roller",
             "apartment_behavior_status": APARTMENT_BEHAVIOR_STATUS[policy.name]}
            for policy in self.policies.values() if policy.command_channels is None
        ]

    @staticmethod
    def validate_command(policy: OfficialPolicy, *, velocity: tuple[float, float, float],
                         posture: str, head: tuple[float, ...], body: tuple[float, ...]) -> None:
        channels = (POLICY_COMMAND_CHANNELS[policy.name] if policy.command_channels is None
                    else policy.command_channels)
        if "twist" not in channels and any(value != 0 for value in velocity):
            raise ValueError(f"{policy.name} does not accept a velocity command")
        if "head" not in channels and any(value != 0 for value in head):
            raise ValueError(f"{policy.name} does not accept a head command")
        if "body" not in channels and any(value != 0 for value in body):
            raise ValueError(f"{policy.name} does not accept a body command")
        if "posture" not in channels and posture != "stand":
            raise ValueError(f"{policy.name} does not accept a posture command")

    @staticmethod
    def command_block(policy: OfficialPolicy, *, velocity: tuple[float, float, float],
                      posture: str, elapsed_s: float, head: tuple[float, ...] = (0.0,) * 4,
                      body: tuple[float, ...] = (0.0,) * 6) -> np.ndarray:
        command = np.zeros(13, dtype=np.float32)
        OfficialPolicyCatalogue.validate_command(policy, velocity=velocity, posture=posture,
                                                 head=head, body=body)
        if policy.encoding == "phase":
            period = float(policy.command["period_s"])
            phase = elapsed_s / period
            command[:2] = (math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase))
        elif policy.encoding == "posture_flag":
            if posture not in {"sit", "stand"}:
                raise ValueError("Posture must be sit or stand")
            command[0] = float(policy.command[posture])
        channels = (POLICY_COMMAND_CHANNELS[policy.name] if policy.command_channels is None
                    else policy.command_channels)
        if policy.encoding == "constant":
            command[:3] = velocity if "twist" in channels else policy.command.get("idle", (0.0,) * 3)
        if "head" in channels:
            command[3:7] = head
        if "body" in channels:
            command[7:13] = body
        return command

    @staticmethod
    def infer(policy: OfficialPolicy, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (61,) or observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise ValueError("Official policy observation must be finite float32[61]")
        action = policy.session.run(None, {policy.session.get_inputs()[0].name: observation[None]})[0]
        if action.shape != (1, 14) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise ValueError("Official policy action must be finite float32[1,14]")
        return action[0].copy()
