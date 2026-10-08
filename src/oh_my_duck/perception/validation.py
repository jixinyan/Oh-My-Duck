import base64
import hashlib
from io import BytesIO
import json
import math

from PIL import Image


def finite_numbers(values) -> bool:
    return all(type(value) in (int, float) and math.isfinite(value) for value in values)


def validate_response(frame: dict, prompt: str, result: dict) -> None:
    json.dumps(result, allow_nan=False)
    for field in ("episode_id", "sequence", "observed_at", "distance_source"):
        if result[field] != frame[field]:
            raise ValueError(f"Perception response does not match frame {field}")
    if result["prompt"] != prompt:
        raise ValueError("Perception response does not match the requested prompt")
    pixels = base64.b64decode(frame["rgb_png_base64"], validate=True)
    if result["image_sha256"] != hashlib.sha256(pixels).hexdigest():
        raise ValueError("Perception response does not match the source image SHA256")
    with Image.open(BytesIO(pixels)) as image:
        image.verify()
        width, height = image.size
    models = result["models"]
    if "yolo26" not in models or not set(models).issubset({"yolo26", "sam3.1"}):
        raise ValueError("Perception response must identify the supported models")
    for digest in models.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Perception model SHA256 is invalid")
    source = result["detection_source"]
    if source not in models:
        raise ValueError("Perception detection source must identify a recorded model")
    identifiers = set()
    for target in result["targets"]:
        identifier = target["target_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Perception target identifiers must be nonempty and unique")
        identifiers.add(identifier)
        if target["detection_source"] != source or target["distance_source"] != frame["distance_source"]:
            raise ValueError("Perception target sources differ from their response")
        if not isinstance(target["label"], str) or not target["label"].strip():
            raise ValueError("Perception target label must be nonempty")
        if not finite_numbers([target["confidence"]]) or not 0 <= target["confidence"] <= 1:
            raise ValueError("Perception confidence must be within zero and one")
        box = target["bbox_xyxy"]
        if (len(box) != 4 or not finite_numbers(box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError("Perception target box must fit the source image")
        count = target["valid_depth_pixels"]
        if type(count) is not int or count < 0 or count > width * height:
            raise ValueError("Perception valid depth count is invalid")
        status = target["distance_status"]
        if status == "insufficient_valid_depth":
            if count >= 8 or any(key in target for key in ("distance_m", "bearing_deg", "surface_position_world_m")):
                raise ValueError("Insufficient depth must not provide a measured target")
        elif status == "valid":
            interval = target["distance_interval_m"]
            position = target["surface_position_world_m"]
            if (not finite_numbers([target["distance_m"], target["distance_xy_m"], target["bearing_deg"]])
                    or not finite_numbers(interval) or not finite_numbers(position)
                    or count < 8 or target["distance_m"] < 0 or target["distance_xy_m"] < 0
                    or len(interval) != 2 or not 0 <= interval[0] <= target["distance_m"] <= interval[1]
                    or len(position) != 3 or not -180 <= target["bearing_deg"] <= 180):
                raise ValueError("Perception target geometry is invalid")
        else:
            raise ValueError("Perception distance status is unsupported")
    annotated = base64.b64decode(result["rgb_png_base64"], validate=True)
    with Image.open(BytesIO(annotated)) as image:
        image.verify()
        if image.size != (width, height):
            raise ValueError("Perception annotation geometry differs from its source image")
