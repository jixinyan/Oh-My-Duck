import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.perception.client import PerceptionClient
from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--prompt", default="desk")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.scene_config.read_text())
    backend = IsaacNewtonBamBackend(robot_id="perception-acceptance", catalog_dir=args.catalog,
        scene_path=root / config["usd_path"], device=config["device"], scene_id=config["scene_id"],
        provenance_path=root / config["provenance_path"], public_map_path=root / config["public_map_path"],
        robot_model=config.get("robot_model", "allcollisions"), observer_renderer=config.get("observer_renderer", "newton_warp"))
    report = {"scope": "actual Newton RGB, segmentation, metric rays and optional model inference", "views": []}
    try:
        for index, yaw in enumerate((0.0, np.pi / 2)):
            pose = {**config["spawn_pose"], "yaw_rad": yaw}
            backend.reset_episode(20260930, config["goal"], pose)
            frame = backend.perception_frame()
            (args.output / f"frame-{index}.json").write_text(json.dumps({key: value for key, value in frame.items()
                if key not in {"rgb_png_base64", "points_world_npy_base64"}}, indent=2) + "\n")
            points = np.load(BytesIO(base64.b64decode(frame["points_world_npy_base64"])), allow_pickle=False)
            rgb = Image.open(BytesIO(base64.b64decode(frame["rgb_png_base64"])))
            assert points.shape == (rgb.height, rgb.width, 3)
            rgb.save(args.output / f"head-{index}.png")
            np.save(args.output / f"points-{index}.npy", points, allow_pickle=False)
            gt = backend.inspect_scene("objects")
            annotated = base64.b64decode(gt.pop("rgb_png_base64"))
            (args.output / f"groundtruth-{index}.png").write_bytes(annotated)
            view = {"yaw_rad": yaw, "ground_truth": gt}
            if args.endpoint:
                result = PerceptionClient(args.endpoint).inspect(frame, args.prompt)
                (args.output / f"models-{index}.png").write_bytes(base64.b64decode(result.pop("rgb_png_base64")))
                view["models"] = result
            report["views"].append(view)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        valid = [target for view in report["views"] for target in view["ground_truth"]["targets"]
                 if target["distance_status"] == "valid"]
        if not valid or not all(np.isfinite(target["distance_m"]) and target["distance_m"] > 0 for target in valid):
            raise AssertionError("Actual visible ground-truth targets lack metric distance")
        if args.endpoint and not any(target["distance_status"] == "valid"
                for view in report["views"] for target in view["models"]["targets"]):
            raise AssertionError("Actual models produced no valid target distance on these views")
        print(json.dumps({"passed": True, "ground_truth_targets": len(valid), "model_inference": bool(args.endpoint)}))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
