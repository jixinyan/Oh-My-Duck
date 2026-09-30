import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import warp as wp

from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.scene_config.read_text())
    backend = IsaacNewtonBamBackend(
        robot_id="camera-acceptance", catalog_dir=args.catalog, scene_path=root / config["usd_path"],
        device=config["device"], scene_id=config["scene_id"], provenance_path=root / config["provenance_path"],
        public_map_path=root / config["public_map_path"])
    try:
        state = backend.reset_episode(20260930, config["goal"], config["spawn_pose"])
        observer = backend.capture_observer()
        report = {"scene": backend.public_scene_info(), "cameras": {}}
        for name, encoded in (("head_camera", state["measurements"]["rgb_png_base64"]),
                              ("observer_camera", observer["rgb_png_base64"])):
            pixels = base64.b64decode(encoded, validate=True)
            (args.output / f"{name}.png").write_bytes(pixels)
            rgb = np.asarray(Image.open(BytesIO(pixels)))
            evidence = backend._render_evidence[name]
            shapes = evidence["visible_shapes"]
            scene_shapes = [shape for shape in shapes if shape["shape"].startswith("/World/Environment/")
                            and "/GroundPlane/" not in shape["shape"] and "/SM_Floor" not in shape["shape"]]
            robot_pixels = sum(shape["pixels"] for shape in shapes if shape["shape"].startswith("/World/envs/"))
            rays = wp.to_torch(backend.native.scene[name]._render_data.camera_rays).cpu().numpy()
            near = backend.native.scene[name].cfg.spawn.clipping_range[0]
            np.testing.assert_allclose(rays[..., 0, 2], -near, atol=1e-7)
            evidence.update(scene_geometry_pixels=sum(shape["pixels"] for shape in scene_shapes),
                            scene_geometry_shapes=len(scene_shapes), robot_pixels=robot_pixels,
                            rgb_bright_fraction=float(np.mean(rgb.max(axis=-1) > 40)))
            report["cameras"][name] = evidence
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        head, observer = report["cameras"]["head_camera"], report["cameras"]["observer_camera"]
        if head["forward_world"][0] < 0.99 or head["up_world"][2] < 0.99:
            raise AssertionError("Head optical axes differ from the official sensor site")
        if head["rgb_std"] < 15 or head["scene_geometry_pixels"] < 100:
            raise AssertionError("Head RGB lacks measurable scene geometry")
        if observer["scene_geometry_shapes"] < 3 or observer["scene_geometry_pixels"] < 1000 or observer["robot_pixels"] < 200:
            raise AssertionError("Observer view lacks the robot and office geometry")
        print(json.dumps({name: {key: value for key, value in evidence.items() if key != "visible_shapes"}
                          for name, evidence in report["cameras"].items()}), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
