import base64
from dataclasses import dataclass
from io import BytesIO
import json

import numpy as np
from PIL import Image

from oh_my_duck.perception.rgbd import validate_pose


@dataclass(frozen=True)
class RGBDFrame:
    pixels: bytes
    image: Image.Image
    points_world_m: np.ndarray
    camera_position_m: np.ndarray
    body_position_m: np.ndarray
    yaw_rad: float


def validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 120:
        raise ValueError("Prompt must contain 1–120 characters")


def decode_frame(frame: dict) -> RGBDFrame:
    json.dumps(frame, allow_nan=False)
    for field in ("episode_id", "observed_at", "distance_source"):
        if not isinstance(frame[field], str) or not frame[field].strip():
            raise ValueError(f"Perception frame {field} must be a nonempty string")
    if type(frame["sequence"]) is not int or frame["sequence"] < 0:
        raise ValueError("Perception frame sequence must be a nonnegative integer")
    pixels = base64.b64decode(frame["rgb_png_base64"], validate=True)
    with Image.open(BytesIO(pixels)) as source:
        if source.format != "PNG" or source.mode != "RGB":
            raise ValueError("Perception RGB must be an RGB PNG image")
        source.load()
        image = source.copy()
    points = np.load(BytesIO(base64.b64decode(frame["points_world_npy_base64"], validate=True)),
                     allow_pickle=False)
    if not isinstance(points, np.ndarray) or points.shape != (image.height, image.width, 3):
        raise ValueError("RGB and world-point geometry must match")
    if points.dtype.kind != "f" or np.isinf(points).any():
        raise ValueError("World points must be floating-point meters without infinities")
    finite = np.isfinite(points)
    if np.any(finite.any(axis=-1) != finite.all(axis=-1)):
        raise ValueError("Missing world points must have three NaN coordinates")
    if type(frame["yaw_rad"]) not in (int, float):
        raise ValueError("Body yaw must be numeric radians")
    for field in ("camera_position_m", "body_position_m"):
        values = frame[field]
        if (not isinstance(values, (list, tuple)) or len(values) != 3
                or any(type(value) not in (int, float) for value in values)):
            raise ValueError(f"Perception frame {field} must contain three numeric meters")
    camera, body = validate_pose(frame["camera_position_m"], frame["body_position_m"], frame["yaw_rad"])
    return RGBDFrame(pixels, image, points, camera, body, float(frame["yaw_rad"]))


def bbox_mask(box, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    values = np.asarray(box, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height)):
        raise ValueError("Perception target box must fit the source image")
    left, top, right, bottom = np.rint(values).astype(int)
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def encode_mask(mask: np.ndarray) -> str:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.dtype(bool):
        raise ValueError("Perception mask must be a two-dimensional boolean array")
    output = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_mask(encoded: str, image_size: tuple[int, int]) -> np.ndarray:
    pixels = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(pixels)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != image_size:
            raise ValueError("Perception mask must be a source-sized grayscale PNG")
        values = np.asarray(image)
        if not np.isin(values, (0, 255)).all():
            raise ValueError("Perception mask PNG must contain only zero and 255")
        return values == 255
