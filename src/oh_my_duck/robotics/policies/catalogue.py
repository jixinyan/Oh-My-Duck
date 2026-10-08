import hashlib
import json
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from oh_my_duck.rl.artifacts.publish.manifest import validate_manifest, ROBOT
from oh_my_duck.robotics.policies.joint_onnx import load_joint_session
from oh_my_duck.robotics.microduck.official_policies import (
    OFFICIAL_REVISION, OfficialPolicy, OfficialPolicyCatalogue,
)


REGISTRY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "policies"],
    "properties": {
        "schema_version": {"const": 1},
        "locomotion": {"type": "object", "additionalProperties": False, "properties": {
            key: {"type": "string", "minLength": 1} for key in ("allcollisions", "groundcontact_rollers")}},
        "policies": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "package_dir", "manifest_sha256", "policy_sha256", "command_channels", "robot_model"],
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"},
                "package_dir": {"type": "string", "minLength": 1},
                "manifest_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "policy_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "command_channels": {"type": "array", "uniqueItems": True,
                                     "items": {"enum": ["twist", "head", "body"]}},
                "robot_model": {"enum": ["allcollisions", "groundcontact_rollers"]},
            },
        }},
    },
}


class PolicyCatalogue(OfficialPolicyCatalogue):
    def __init__(self, directory: Path, registry_path: Path | None = None):
        super().__init__(directory)
        self.revision = OFFICIAL_REVISION
        self._packages: list[dict] = []
        self._locomotion = {"allcollisions": "alpha_walking", "groundcontact_rollers": "roller"}
        if registry_path is None:
            return
        registry_path = registry_path.resolve(strict=True)
        registry_bytes = registry_path.read_bytes()
        registry = json.loads(registry_bytes)
        Draft202012Validator(REGISTRY_SCHEMA).validate(registry)
        digest = hashlib.sha256(registry_bytes).hexdigest()
        self.revision += "+registry:" + digest
        for entry in registry["policies"]:
            name = entry["name"]
            if name in self.policies:
                raise ValueError(f"Policy alias already exists: {name}")
            package = (registry_path.parent / entry["package_dir"]).resolve(strict=True)
            manifest_path, policy_path = package / "manifest.json", package / "policy.onnx"
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != entry["manifest_sha256"]:
                raise ValueError(f"Registered manifest SHA256 differs: {name}")
            if hashlib.sha256(policy_path.read_bytes()).hexdigest() != entry["policy_sha256"]:
                raise ValueError(f"Registered policy SHA256 differs: {name}")
            if sorted(path.name for path in package.glob("*.onnx")) != ["policy.onnx"]:
                raise ValueError(f"Registered package must contain one policy.onnx: {name}")
            manifest = json.loads(manifest_path.read_text())
            validate_manifest(manifest)
            if (manifest.get("schema_version"), manifest.get("model_api"), manifest.get("obs_len"),
                    manifest.get("action_len"), manifest.get("robot")) != (2, 1, 61, 14, ROBOT):
                raise ValueError(f"Registered package requires schema 2, API 1 and the Microduck robot: {name}")
            if "policies" in manifest or manifest.get("kind") not in {"perpetual", "episodic"}:
                raise ValueError(f"Registered package requires one perpetual or episodic policy: {name}")
            command = manifest["command"]
            if command.get("encoding") != "constant":
                raise ValueError(f"Registered package requires constant command encoding: {name}")
            idle = command.get("idle")
            if (not isinstance(idle, list) or len(idle) != 3 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in idle)):
                raise ValueError(f"Registered package requires a finite idle twist: {name}")
            if any(value != 0 for value in idle):
                raise ValueError(f"Native registered policy requires zero idle twist: {name}")
            duration = manifest.get("duration_s")
            if manifest["kind"] == "perpetual" and duration is not None:
                raise ValueError(f"Perpetual policy cannot declare a duration: {name}")
            if manifest["kind"] == "episodic" and (isinstance(duration, bool) or
                    not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Episodic policy requires a positive finite duration: {name}")
            scale = manifest.get("action_scale", 1.0)
            if isinstance(scale, bool) or scale != 1.0:
                raise ValueError(f"Registered policy must preserve the native action scale 1.0: {name}")
            if "chain" in manifest and not isinstance(manifest["chain"], bool):
                raise ValueError(f"Registered policy chain must be boolean: {name}")
            for key in ("ramp_s", "unwind_s"):
                value = manifest.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Registered policy {key} must be positive and finite: {name}")
            for key in ("name", "entry_pose", "slot"):
                if key in manifest and (not isinstance(manifest[key], str) or not manifest[key].strip()):
                    raise ValueError(f"Registered policy {key} must be a nonempty string: {name}")
            session = load_joint_session(policy_path)
            validation = manifest.get("validation", {})
            if "policy_sha256" in validation and validation["policy_sha256"] != entry["policy_sha256"]:
                raise ValueError(f"Package validation policy SHA256 differs: {name}")
            policy = OfficialPolicy(name, str(policy_path), entry["policy_sha256"], manifest["kind"],
                "constant", 1.0, duration, "roller" if entry["robot_model"] == "groundcontact_rollers" else "walk",
                manifest.get("chain", False), manifest.get("slot"), manifest.get("entry_pose"),
                manifest.get("ramp_s"), manifest.get("unwind_s"), command, session,
                tuple(entry["command_channels"]))
            self.infer(policy, np.zeros(61, dtype=np.float32))
            self.policies[name] = policy
            self._packages.append({"name": name, "file": str(policy_path), "sha256": policy.sha256,
                "kind": policy.kind, "encoding": policy.encoding, "action_scale": policy.action_scale,
                "duration_s": duration, "entry_pose": policy.entry_pose, "slot": policy.slot,
                "chain": policy.chain, "ramp_s": policy.ramp_s, "unwind_s": policy.unwind_s,
                "command": command, "command_channels": list(policy.command_channels),
                "required_robot_mode": entry["robot_model"], "executable_here": policy.mode != "roller",
                "source": "registered_schema2_package", "package_name": manifest.get("name"),
                "manifest_sha256": entry["manifest_sha256"], "registry_sha256": digest,
                "training": manifest.get("training", {}), "validation": validation,
                "apartment_behavior_status": "requires_independent_behavior_acceptance"})
        for model, name in registry.get("locomotion", {}).items():
            policy = self.get(name)
            if (policy.kind != "perpetual" or "twist" not in (
                    policy.command_channels if policy.command_channels is not None else
                    ("twist",) if name in {"alpha_walking", "velstand", "roller"} else ()) or
                    (policy.mode == "roller") != (model == "groundcontact_rollers")):
                raise ValueError(f"Locomotion policy requires matching robot mode and perpetual twist commands: {name}")
            self._locomotion[model] = name

    def describe_apartment(self) -> list[dict]:
        return super().describe_apartment() + [dict(item) for item in self._packages]

    def locomotion_policy(self, robot_model: str) -> str:
        return self._locomotion[robot_model]
