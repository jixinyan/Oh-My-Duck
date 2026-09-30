from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from oh_my_duck.rl.artifacts.inference import cpu_session
from oh_my_duck.rl.artifacts.publish.manifest import check_onnx
from oh_my_duck.rl.backends.isaac_newton.contracts import OBSERVATION_NAMES
from oh_my_duck.robotics.microduck.protocol import HOME, JOINT_NAMES


OFFICIAL_REPO = "pollen-robotics/microduck-policies"
OFFICIAL_REVISION = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"
MANIFEST_SHA256 = "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94"
POLICY_SHA256 = {
    "alpha_ground_pick.onnx": "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
    "alpha_sitstand.onnx": "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    "alpha_walking.onnx": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    "ball_kick_left.onnx": "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
    "roller.onnx": "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd",
    "roller_crouch.onnx": "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
    "roulade.onnx": "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b",
    "velstand.onnx": "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45",
}

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
            shape = check_onnx(path)
            if shape.recurrent or shape.obs_len != 61 or shape.action_len != 14:
                raise ValueError(f"Official API-1 policy shape differs: {file}")
            session = cpu_session(path)
            if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                raise ValueError(f"Official policy tensor count differs: {file}")
            if session.get_inputs()[0].type != "tensor(float)" or session.get_outputs()[0].type != "tensor(float)":
                raise ValueError(f"Official policy tensor type differs: {file}")
            metadata = session.get_modelmeta().custom_metadata_map
            if tuple(metadata.get("joint_names", "").split(",")) != JOINT_NAMES:
                raise ValueError(f"Official policy joint order differs: {file}")
            if tuple(metadata.get("observation_names", "").split(",")) != OBSERVATION_NAMES:
                raise ValueError(f"Official policy observation order differs: {file}")
            home = np.asarray([float(value) for value in metadata.get("default_joint_pos", "").split(",")])
            if home.shape != (14,) or not np.allclose(home, HOME, atol=5e-4, rtol=0):
                raise ValueError(f"Official policy HOME differs: {file}")
            if float(metadata.get("action_scale", "nan")) != 1.0:
                raise ValueError(f"Official policy graph action scale differs: {file}")
            value = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
            if value.shape != (1, 14) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Official policy CPU inference failed: {file}")
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
            for policy in self.policies.values()
        ]

    @staticmethod
    def validate_command(policy: OfficialPolicy, *, velocity: tuple[float, float, float],
                         posture: str, head: tuple[float, ...], body: tuple[float, ...]) -> None:
        channels = POLICY_COMMAND_CHANNELS[policy.name]
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
        elif policy.kind == "perpetual" and policy.name in {"alpha_walking", "velstand", "roller"}:
            command[:3] = velocity
        channels = POLICY_COMMAND_CHANNELS[policy.name]
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
