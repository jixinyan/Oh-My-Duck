import argparse
import base64
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def native_position(export: Path) -> list[float]:
    events = json.loads((export / "source" / "events.json").read_text())
    checks = [event for event in events if event["type"] == "verification.checked"]
    fact = next(item for item in checks[-1]["detail"]["facts"]
                if item["check_id"] == "goal_reached")
    if fact["value"] is not True:
        raise ValueError("The source run must contain a passed native goal check")
    return json.loads(fact["reason"])["evidence"]["robot_world_position_m"]


def run(export: Path, output: Path) -> dict:
    position = native_position(export)
    root = Path(__file__).resolve().parents[1]
    backend = CpuMujocoBamBackend(robot_id="observer-yaw-acceptance",
                                  catalog_dir=root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    orientations = []
    try:
        for index in range(16):
            angle_deg = index * 22.5
            yaw_rad = math.radians(angle_deg)
            backend.reset_episode(seed=20260929,
                                  goal={"kind": "room", "room": "office", "hold_ticks": 5},
                                  spawn_pose={"x_m": position[0], "y_m": position[1],
                                              "yaw_rad": yaw_rad})
            frame = backend.capture_observer(include_segmentation=True)
            png = base64.b64decode(frame["rgb_png_base64"])
            path = output / f"observer-yaw-{index:02d}.png"
            path.write_bytes(png)
            with Image.open(BytesIO(png)) as image:
                if image.format != "PNG" or image.size != (640, 480):
                    raise ValueError("Observer frame format or dimensions differ")
                variance = float(np.asarray(image.convert("RGB")).var())
            segmentation = frame["segmentation"]
            names = set(segmentation["visible_robot_body_names"])
            if (segmentation["visible_robot_geom_pixels"] < 1000 or
                    segmentation["visible_apartment_geom_pixels"] < 1000 or
                    "trunk_base" not in names or "neck" not in names or
                    variance < 1):
                raise ValueError(f"Observer view is obscured at yaw {angle_deg} degrees")
            orientations.append({"yaw_deg": angle_deg, "yaw_rad": yaw_rad,
                                 "camera": frame["camera"], "png": str(path),
                                 "robot_pixels": segmentation["visible_robot_geom_pixels"],
                                 "apartment_pixels": segmentation["visible_apartment_geom_pixels"],
                                 "visible_robot_body_names": segmentation["visible_robot_body_names"],
                                 "pixel_variance": variance})
    finally:
        backend.close()
    return {"source_run_id": export.name, "source_position_m": position,
            "source_yaw_recorded": False, "tested_orientations": orientations,
            "minimum_robot_pixels": min(item["robot_pixels"] for item in orientations),
            "status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.success_export, args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "source_run_id": result["source_run_id"],
                      "orientation_count": len(result["tested_orientations"]),
                      "minimum_robot_pixels": result["minimum_robot_pixels"]}, indent=2))


if __name__ == "__main__":
    main()
