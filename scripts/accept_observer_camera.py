import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def frame_evidence(frame: dict, output: Path, name: str) -> dict:
    png = base64.b64decode(frame["rgb_png_base64"])
    (output / f"{name}.png").write_bytes(png)
    with Image.open(BytesIO(png)) as image:
        if image.format != "PNG" or image.size != (640, 480):
            raise ValueError("Observer camera image format or dimensions differ")
        pixels = np.asarray(image.convert("RGB"))
    if pixels.shape != (480, 640, 3) or float(pixels.var()) < 1:
        raise ValueError("Observer camera image pixels are uniform")
    segmentation = frame["segmentation"]
    evidence = {"episode_id": frame["episode_id"], "sequence": frame["sequence"],
                "simulation_time_s": frame["simulation_time_s"], "camera": frame["camera"],
                "png_bytes": len(png), "pixel_min": int(pixels.min()), "pixel_max": int(pixels.max()),
                "pixel_variance": float(pixels.var()), "segmentation": segmentation}
    (output / f"{name}-metadata.json").write_text(json.dumps(evidence, indent=2) + "\n")
    required_bodies = {"trunk_base", "neck", "upper_leg_left", "upper_leg_right",
                       "ankle_left", "ankle_right"}
    if (segmentation["visible_robot_geom_pixels"] < 1000 or
            segmentation["visible_robot_geom_count"] < 3 or
            segmentation["visible_apartment_geom_pixels"] < 1000 or
            not required_bodies.issubset(segmentation["visible_robot_body_names"])):
        raise ValueError("Observer camera did not expose the robot body and apartment")
    return evidence


def run(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    backend = CpuMujocoBamBackend(robot_id="observer-camera-acceptance",
                                  catalog_dir=root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    try:
        observation = backend.reset_episode(seed=42, goal={"kind": "room", "room": "corridor",
                                                            "hold_ticks": 5})
        first = frame_evidence(backend.capture_observer(include_segmentation=True), output, "observer_initial")
        if first["sequence"] != observation["sequence"] or first["simulation_time_s"] != 0:
            raise ValueError("Observer capture changed the initial physical sequence")
        for index in range(50):
            inference = backend.infer_policy()
            backend.apply_policy_action(inference["action"], request_id=f"prelude:{index}",
                                        expected_sequence=inference["sequence"], should_stop=lambda: False)
        backend.select_policy("alpha_walking", request_id="select:alpha_walking")
        backend.set_command({"twist": [0.3, 0.0, 0.0]}, request_id="walk:command")
        for index in range(50):
            inference = backend.infer_policy()
            backend.apply_policy_action(inference["action"], request_id=f"walk:{index}",
                                        expected_sequence=inference["sequence"], should_stop=lambda: False)
        final = frame_evidence(backend.capture_observer(include_segmentation=True), output, "observer_after_motion")
        if final["sequence"] != 100 or final["simulation_time_s"] <= first["simulation_time_s"]:
            raise ValueError("Observer capture does not reference current physical motion")
        head = backend.observe_control()["measurements"]
        if head["rgb_width"] != 320 or head["rgb_height"] != 240 or head["camera_frame_id"] != "head_camera":
            raise ValueError("Observer capture changed native head camera output")
        return {"status": "passed", "initial": first, "after_motion": final,
                "head_camera": {"width": head["rgb_width"], "height": head["rgb_height"],
                                "frame_id": head["camera_frame_id"]}}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
