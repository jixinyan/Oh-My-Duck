import argparse
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from oh_my_duck.perception.client import PerceptionClient
from oh_my_duck.perception.frames import decode_frame, decode_mask, encode_mask, validate_prompt
from oh_my_duck.perception.validation import validate_measurements, validate_response


def _save(path: Path, value: dict) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def _reject(operation) -> None:
    try:
        operation()
    except (ValueError, KeyError):
        return
    raise AssertionError("Altered perception measurements were accepted")


def audit_capture(capture: Path) -> dict:
    record = json.loads((capture / "capture.json").read_text())
    if not record["cases"]:
        raise AssertionError("Perception capture requires actual frames")
    targets, rejections, hashes = 0, 0, {}
    for case in record["cases"]:
        for field in ("frame", "response"):
            name = case[field]
            if Path(name).name != name:
                raise ValueError("Perception capture filenames must remain within its directory")
            digest = hashlib.sha256((capture / name).read_bytes()).hexdigest()
            if digest != case[field + "_sha256"]:
                raise AssertionError("Perception capture source SHA256 differs")
            hashes[name] = digest
        frame = json.loads((capture / case["frame"]).read_text())
        result = json.loads((capture / case["response"]).read_text())
        source = decode_frame(frame)
        validate_response(frame, record["prompt"], result)
        validate_measurements(frame, result, decoded=source)
        if result["models"] != record["health"]["models"]:
            raise AssertionError("Perception response model identity differs from service health")
        for index, target in enumerate(result["targets"]):
            if target["mask_source"] == "yolo26_bbox":
                left, top, right, bottom = np.rint(target["bbox_xyxy"]).astype(int)
                mask = np.zeros(source.points_world_m.shape[:2], dtype=bool)
                mask[top:bottom, left:right] = True
            else:
                mask = decode_mask(target["mask_png_base64"], source.image.size)
            # 使用原始像素独立计算距离、分位数、位置和方向。
            selected = source.points_world_m[mask & np.isfinite(source.points_world_m).all(axis=-1)]
            if target["valid_depth_pixels"] != len(selected):
                raise AssertionError("Measured pixel count differs from original world points")
            if len(selected) >= 8:
                center = np.median(selected, axis=0)
                distances = np.linalg.norm(selected - source.camera_position_m, axis=-1)
                dx, dy = center[:2] - source.body_position_m[:2]
                angle = math.atan2(float(dy), float(dx)) - source.yaw_rad
                expected = {"distance_m": np.median(distances),
                            "distance_interval_m": np.quantile(distances, [0.1, 0.9]),
                            "distance_xy_m": math.hypot(float(dx), float(dy)),
                            "bearing_deg": math.degrees(math.atan2(math.sin(angle), math.cos(angle))),
                            "surface_position_world_m": center}
                for field, value in expected.items():
                    np.testing.assert_allclose(target[field], value, rtol=0, atol=1e-6)
                alterations = [("distance_m", target["distance_m"] + 0.01),
                               ("distance_xy_m", target["distance_xy_m"] + 0.01),
                               ("bearing_deg", target["bearing_deg"] + 0.01),
                               ("distance_interval_m", [value + 0.01 for value in target["distance_interval_m"]]),
                               ("surface_position_world_m", [value + 0.01 for value in target["surface_position_world_m"]]),
                               ("valid_depth_pixels", target["valid_depth_pixels"] + 1),
                               ("distance_status", "insufficient_valid_depth"),
                               ("mask_source", "unsupported")]
                for field, value in alterations:
                    altered = deepcopy(result)
                    altered["targets"][index][field] = value
                    _reject(lambda: validate_measurements(frame, altered, decoded=source))
                    rejections += 1
            encoded = encode_mask(mask)
            if not np.array_equal(decode_mask(encoded, source.image.size), mask):
                raise AssertionError("Original target pixel mask changed during PNG encoding")
            _reject(lambda: decode_mask(encoded, (source.image.width + 1, source.image.height)))
            rejections += 1
            targets += 1
        for field, value in (("sequence", True), ("episode_id", ""), ("observed_at", ""),
                             ("camera_position_m", [float("inf"), 0, 0]), ("yaw_rad", True)):
            altered = deepcopy(frame)
            altered[field] = value
            _reject(lambda: decode_frame(altered))
            rejections += 1
        for kind in ("infinite", "partial_nan", "wrong_shape", "integer"):
            altered = deepcopy(frame)
            points = source.points_world_m.copy()
            if kind == "infinite":
                points[0, 0] = [float("inf")] * 3
            elif kind == "partial_nan":
                points[0, 0] = [float("nan"), 0, 0]
            elif kind == "wrong_shape":
                points = points[:, :-1]
            else:
                points = np.zeros(points.shape, dtype=np.int32)
            buffer = BytesIO()
            np.save(buffer, points, allow_pickle=False)
            altered["points_world_npy_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            _reject(lambda: decode_frame(altered))
            rejections += 1
    if targets == 0:
        raise AssertionError("Perception measurement acceptance requires actual model targets")
    hashes["capture.json"] = hashlib.sha256((capture / "capture.json").read_bytes()).hexdigest()
    return {"passed": True, "scope": "Recorded model targets and independent original RGBD measurement audit",
            "cases": len(record["cases"]), "model_targets": targets, "invalid_measurements_rejected": rejections,
            "input_sha256": hashes, "device": record["health"]["device"],
            "models": record["health"]["models"], "new_model_inference": False}


def main():
    parser = argparse.ArgumentParser(description="执行模型感知或复核原始 RGBD 测量记录")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frames", type=Path, help="Directory containing original *-frame.json files")
    inputs.add_argument("--capture", type=Path, help="Directory containing a saved capture.json")
    parser.add_argument("--endpoint", help="Explicit local model service or SSH tunnel")
    parser.add_argument("--prompt", default="objects")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.frames is None) != (args.endpoint is None):
        parser.error("Model inference requires --frames and --endpoint together")
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_prompt(args.prompt)
    if args.capture is not None:
        result = audit_capture(args.capture.resolve(strict=True))
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        client = PerceptionClient(args.endpoint)
        frames = sorted(args.frames.resolve(strict=True).glob("*-frame.json"))
        if not frames:
            raise ValueError("Original RGBD frame files are required")
        with urlopen(client.endpoint + "/health", timeout=30) as response:
            health = json.load(response)
        if health["measurement_validation"] != "source_rgbd":
            raise ValueError("Perception service must validate original RGBD measurements")
        args.output.mkdir(parents=True, exist_ok=False)
        cases = []
        for index, path in enumerate(frames):
            source = path.read_bytes()
            frame = json.loads(source)
            result = client.inspect(frame, args.prompt)
            frame_name, response_name = f"case-{index}-frame.json", f"case-{index}-response.json"
            with (args.output / frame_name).open("xb") as output:
                output.write(source)
            _save(args.output / response_name, result)
            cases.append({"source_frame": str(path), "frame": frame_name, "response": response_name,
                          "frame_sha256": hashlib.sha256(source).hexdigest(),
                          "response_sha256": hashlib.sha256((args.output / response_name).read_bytes()).hexdigest()})
        _save(args.output / "capture.json", {"prompt": args.prompt, "health": health, "cases": cases})
        result = audit_capture(args.output)
        result["new_model_inference"] = True
    _save(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "cases", "model_targets", "invalid_measurements_rejected",
                                                  "device", "new_model_inference")}))


if __name__ == "__main__":
    main()
