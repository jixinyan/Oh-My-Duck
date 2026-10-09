import base64
import hashlib
from io import BytesIO
import json
import math

from PIL import Image
import numpy as np

from oh_my_duck.perception.frames import RGBDFrame, bbox_mask, decode_frame, decode_mask
from oh_my_duck.perception.rgbd import measure_target


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
            if count >= 8 or any(key in target for key in (
                    "distance_m", "distance_xy_m", "distance_interval_m", "bearing_deg", "surface_position_world_m")):
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


def validate_measurements(frame: dict, result: dict, *, decoded: RGBDFrame | None = None) -> None:
    source = decode_frame(frame) if decoded is None else decoded
    for target in result["targets"]:
        if target["detection_source"] == "yolo26" and target["mask_source"] == "yolo26_bbox":
            if "mask_png_base64" in target:
                raise ValueError("YOLO bbox measurements must use the recorded bounding box")
            mask = bbox_mask(target["bbox_xyxy"], source.image.size)
        elif target["detection_source"] == target["mask_source"] == "sam3.1":
            mask = decode_mask(target["mask_png_base64"], source.image.size)
            rows, columns = np.nonzero(mask)
            if len(rows) < 8:
                raise ValueError("SAM targets require at least eight mask pixels")
            expected_box = [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]
            if target["bbox_xyxy"] != expected_box:
                raise ValueError("SAM target box differs from the recorded mask")
        else:
            raise ValueError("Perception detection and mask sources are inconsistent")
        expected = measure_target(mask, source.points_world_m, source.camera_position_m,
                                  source.body_position_m, source.yaw_rad)
        if (target["distance_status"] != expected["distance_status"]
                or target["valid_depth_pixels"] != expected["valid_depth_pixels"]):
            raise ValueError("Perception target depth status differs from its source pixels")
        for field in ("distance_m", "distance_xy_m", "distance_interval_m", "bearing_deg", "surface_position_world_m"):
            if field in expected and not np.allclose(target[field], expected[field], rtol=0, atol=1e-6):
                raise ValueError(f"Perception target {field} differs from its source RGBD")
