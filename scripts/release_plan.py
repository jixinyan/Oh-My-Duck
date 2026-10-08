import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from accept_metric_campaign import PLAN_SCHEMA
from metric_plan import case_motions


STAGE_SCHEMA = {
    "type": "object", "required": ["schema_version", "stages"], "additionalProperties": False,
    "properties": {"schema_version": {"const": 1}, "stages": {"type": "array", "minItems": 1,
        "items": {"oneOf": [
            {"type": "object", "required": ["id", "kind", "plan", "suite"], "additionalProperties": False,
             "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"},
                 "kind": {"const": "metric"}, "plan": {"type": "string", "minLength": 1},
                 "suite": {"type": "string", "minLength": 1}}},
            {"type": "object", "required": ["id", "kind", "scene", "instruction", "minimum_distance_m"],
             "additionalProperties": False, "properties": {
                 "id": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"}, "kind": {"const": "navigation"},
                 "scene": {"type": "string", "minLength": 1}, "instruction": {"type": "string", "minLength": 1},
                 "minimum_distance_m": {"type": "number", "exclusiveMinimum": 0}}},
        ]}}},
}
POINT_SCHEMA = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
SCENE_SCHEMA = {
    "type": "object", "required": ["backend", "scene_id", "usd_path", "provenance_path", "public_map_path",
        "spawn_pose", "goal", "task_instruction", "budget"],
    "properties": {
        "backend": {"const": "isaac-newton"}, "scene_id": {"type": "string", "minLength": 1},
        **{key: {"type": "string", "minLength": 1} for key in ("usd_path", "provenance_path", "public_map_path")},
        "device": {"const": "cuda:0"},
        "observer_renderer": {"enum": ["newton_warp", "isaac_rtx"]},
        "robot_model": {"enum": ["allcollisions", "groundcontact_rollers"]},
        "spawn_pose": {"type": "object", "required": ["x_m", "y_m", "z_m", "yaw_rad"],
            "additionalProperties": False, "properties": {
                **{key: {"type": "number"} for key in ("x_m", "y_m", "yaw_rad")},
                "z_m": {"type": "number", "exclusiveMinimum": 0}}},
        "goal": {"type": "object", "required": ["kind", "target_xy_m", "distance_m", "hold_ticks"],
            "properties": {"kind": {"const": "point"}, "target_xy_m": POINT_SCHEMA,
                "distance_m": {"type": "number", "exclusiveMinimum": 0},
                "hold_ticks": {"type": "integer", "minimum": 5},
                "waypoints": {"type": "array", "items": {"type": "object",
                    "required": ["target_xy_m", "distance_m"], "properties": {
                        "target_xy_m": POINT_SCHEMA, "distance_m": {"type": "number", "exclusiveMinimum": 0}}}}}},
        "task_instruction": {"type": "string", "minLength": 1},
        "budget": {"type": "object", "required": ["max_control_steps", "max_wall_time_s"],
            "additionalProperties": False, "properties": {
                "max_control_steps": {"type": "integer", "minimum": 5},
                "max_wall_time_s": {"type": "number", "exclusiveMinimum": 0}}},
    },
}


def within(root, name):
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Acceptance input paths must be relative to the source checkout")
    path = (root / candidate).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Acceptance input must be a file in the source checkout")
    return path


def validate_plan(root, path):
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Acceptance plan must belong to the source checkout")
    plan = json.loads(path.read_text())
    json.dumps(plan, allow_nan=False)
    Draft202012Validator(STAGE_SCHEMA).validate(plan)
    if len({stage["id"] for stage in plan["stages"]}) != len(plan["stages"]):
        raise ValueError("Acceptance stage identities must be unique")
    scenes, inputs = {}, {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}
    for stage in plan["stages"]:
        if stage["kind"] == "metric":
            metric_path = within(root, stage["plan"])
            inputs[stage["plan"]] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
            metrics = json.loads(metric_path.read_text())
            json.dumps(metrics, allow_nan=False)
            Draft202012Validator(PLAN_SCHEMA).validate(metrics)
            if stage["suite"] not in metrics["suites"]:
                raise ValueError("Unknown metric suite")
            suite = metrics["suites"][stage["suite"]]
            cases = suite["cases"]
            if len({case["id"] for case in cases}) != len(cases):
                raise ValueError("Metric case identities must be unique")
            for case in cases:
                case_motions(case)
            scene_name = suite["scene_config"]
        else:
            instruction = within(root, stage["instruction"])
            if not instruction.read_text().strip():
                raise ValueError("Navigation instruction must be nonempty")
            inputs[stage["instruction"]] = hashlib.sha256(instruction.read_bytes()).hexdigest()
            scene_name = stage["scene"]
        scene_path = within(root, scene_name)
        configuration = json.loads(scene_path.read_text())
        json.dumps(configuration, allow_nan=False)
        Draft202012Validator(SCENE_SCHEMA).validate(configuration)
        for key in ("usd_path", "provenance_path", "public_map_path"):
            asset = Path(configuration[key])
            if asset.is_absolute() or ".." in asset.parts:
                raise ValueError("Scene assets must use project-relative paths without parent components")
        inputs[scene_name] = hashlib.sha256(scene_path.read_bytes()).hexdigest()
        scenes[scene_name] = configuration
    return plan, scenes, inputs
