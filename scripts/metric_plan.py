import math

from jsonschema import Draft202012Validator


MOTIONS_SCHEMA = {
    "type": "array", "minItems": 1, "items": {"oneOf": [
        {"type": "object", "required": ["operation", "arguments"], "additionalProperties": False,
         "properties": {"operation": {"const": "walk"}, "arguments": {
             "type": "object", "required": ["distance_m"], "additionalProperties": False,
             "properties": {"distance_m": {"type": "number", "minimum": -10, "maximum": 10},
                 "speed_m_s": {"type": "number", "minimum": 0.1, "maximum": 0.4}}}}},
        {"type": "object", "required": ["operation", "arguments"], "additionalProperties": False,
         "properties": {"operation": {"const": "rotate"}, "arguments": {
             "type": "object", "required": ["angle_deg"], "additionalProperties": False,
             "properties": {"angle_deg": {"type": "number", "minimum": -360, "maximum": 360},
                 "angular_speed_deg_s": {"type": "number", "minimum": 10, "maximum": 55}}}}},
    ]},
}


def case_motions(case):
    if "motions" in case:
        motions = case["motions"]
    else:
        arguments = {"walk": {"distance_m": case["distance_m"], "speed_m_s": case.get("speed_m_s", 0.4)},
                     "rotate": {"angle_deg": case["angle_deg"], "angular_speed_deg_s": case.get("angular_speed_deg_s", 45)}}
        motions = [{"operation": operation, "arguments": arguments[operation]} for operation in case["operations"]]
    Draft202012Validator(MOTIONS_SCHEMA).validate(motions)
    normalized = []
    for motion in motions:
        operation, arguments = motion["operation"], dict(motion["arguments"])
        key, speed_key, default, minimum = (("distance_m", "speed_m_s", 0.4, 0.1) if operation == "walk"
                                          else ("angle_deg", "angular_speed_deg_s", 45, 10))
        if any(not math.isfinite(value) for value in arguments.values()) or abs(arguments[key]) < minimum:
            raise ValueError("Motion parameters must be finite and meet the minimum movement magnitude")
        arguments.setdefault(speed_key, default)
        normalized.append({"operation": operation, "arguments": arguments})
    return normalized
