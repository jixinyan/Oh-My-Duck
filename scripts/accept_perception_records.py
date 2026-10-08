import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.perception.client import PerceptionClient
from oh_my_duck.perception.rgbd import measure_target
from oh_my_duck.perception.validation import validate_response


def require_rejection(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid perception data was accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = args.capture.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((capture / "result.json").read_text())
    views, targets, rejection_checks, hashes = 0, 0, 0, {}
    for index, view in enumerate(report["views"]):
        result = deepcopy(view["models"])
        source = (capture / f"head-{index}.png").read_bytes()
        result["rgb_png_base64"] = base64.b64encode((capture / f"models-{index}.png").read_bytes()).decode()
        frame = {field: result[field] for field in ("episode_id", "sequence", "observed_at", "distance_source")}
        frame["rgb_png_base64"] = base64.b64encode(source).decode()
        validate_response(frame, result["prompt"], result)
        for field, value in (("episode_id", "different-episode"), ("sequence", result["sequence"] + 1),
                             ("observed_at", "different-time"), ("distance_source", "different-source"),
                             ("prompt", "different-prompt"), ("image_sha256", "0" * 64)):
            invalid = deepcopy(result)
            invalid[field] = value
            require_rejection(lambda: validate_response(frame, result["prompt"], invalid))
            rejection_checks += 1
        points = np.load(capture / f"points-{index}.npy", allow_pickle=False)
        with Image.open(capture / f"head-{index}.png") as image:
            assert points.shape == (image.height, image.width, 3)
        for target in result["targets"]:
            left, top, right, bottom = np.rint(target["bbox_xyxy"]).astype(int)
            mask = np.zeros(points.shape[:2], dtype=bool)
            mask[max(top, 0):min(bottom, mask.shape[0]), max(left, 0):min(right, mask.shape[1])] = True
            selected = points[mask & np.isfinite(points).all(axis=-1)]
            assert len(selected) == target["valid_depth_pixels"]
            if target["distance_status"] == "valid":
                np.testing.assert_allclose(np.median(selected, axis=0), target["surface_position_world_m"],
                                           rtol=0, atol=1e-6)
                for field, value in (("distance_m", float("nan")), ("confidence", 2),
                                     ("bbox_xyxy", [-1, 0, 10, 10]), ("valid_depth_pixels", 0),
                                     ("surface_position_world_m", ["invalid", "invalid", "invalid"])):
                    invalid = deepcopy(result)
                    invalid["targets"][result["targets"].index(target)][field] = value
                    require_rejection(lambda: validate_response(frame, result["prompt"], invalid))
                    rejection_checks += 1
            # 从实际 RGBD 检查无效 pose 在距离计算前被拒绝。
            empty = np.zeros_like(mask)
            require_rejection(lambda: measure_target(empty, points, [float("nan")] * 3,
                                                      [0.0] * 3, view["yaw_rad"]))
            require_rejection(lambda: measure_target(empty, points, [0.0] * 3,
                                                      [0.0] * 3, float("inf")))
            rejection_checks += 2
            targets += 1
        for name in ("result.json", f"head-{index}.png", f"models-{index}.png", f"points-{index}.npy"):
            hashes[name] = hashlib.sha256((capture / name).read_bytes()).hexdigest()
        views += 1
    if views == 0 or targets == 0:
        raise AssertionError("Recorded model detections are required")
    for endpoint in ("http://127.0.0.1:8784", "http://127.0.0.1:8784/"):
        assert PerceptionClient(endpoint).endpoint == "http://127.0.0.1:8784"
    for endpoint in ("http://127.0.0.1:8784@remote.example", "http://user@127.0.0.1:8784",
                     "http://127.0.0.1:0", "http://127.0.0.1:8784/path", "http://127.0.0.1:8784?q=1",
                     "http://127.0.0.1:8784#fragment", "https://127.0.0.1:8784", "http://127.0.0.1"):
        require_rejection(lambda: PerceptionClient(endpoint))
        rejection_checks += 1
    acceptance = {"passed": True, "scope": "Recorded actual Newton RGBD and YOLO response validation",
                  "views": views, "model_targets": targets, "invalid_data_rejected": rejection_checks,
                  "input_sha256": hashes, "new_model_inference": False, "gpu_acceptance_performed": False}
    args.output.write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance))


if __name__ == "__main__":
    main()
