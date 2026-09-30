import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def read(root, name):
    return json.loads((root / f"{name}.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--visuals", type=Path, required=True)
    args = parser.parse_args()
    result, samples, frames = [read(args.evidence, name) for name in ("result", "samples", "frames")]
    render = read(args.visuals, "render")
    total = result["finish"]["execution"]["control_steps"]
    if ([s["sequence"] for s in samples] != list(range(total+1)) or
            result["runtime"]["physics_calls"] != total*4 or
            result["finish"]["execution"]["raw_sim_steps"] != total*4 or
            result["guard"]["reason"] != "forward_proximity" or
            result["initial_central_tof_closest_mm"] < 90 or
            result["guard"]["central_tof_closest_mm"] >= 90):
        raise AssertionError("Native control coverage or proximity threshold evidence differs")
    for sample in samples:
        if (len(sample["tof_distance_mm"]) != 64 or len(sample["tof_status"]) != 64 or
                not np.isfinite([*sample["body_position_m"], *sample["body_twist"], *sample["imu"]]).all()
                or sample["contact_evidence"]["non_ground_external_contact_samples_total"]):
            raise AssertionError("Sensor evidence is invalid or contains external contact")
    zero = [s for s in samples if s["phase"] == "stop"]
    if len(zero) != result["finish"]["stop_confirmation"]["zero_control_steps"]:
        raise AssertionError("Zero-command sample count differs")
    for sample in zero[-5:]:
        if max(abs(v) for v in sample["body_twist"][:2]) > .025 or abs(sample["body_twist"][2]) > .08:
            raise AssertionError("Final five measured samples exceed stop thresholds")
    wall = result["configuration"]["expected_wall"]
    low, high = np.asarray(wall["min"]), np.asarray(wall["max"])
    hit_count = 0
    for label in ("initial", "guard", "final"):
        state = read(args.evidence, "sensor-" + label)["measurements"]
        origin = np.asarray(state["tof_ray_origin_world_m"])
        for distance, direction, name in zip(state["tof_hit_distance_m"], state["tof_ray_directions_world"],
                                             state["tof_hit_geoms"], strict=True):
            if name and wall["path"].removeprefix("/Root/") in name:
                point = origin + distance*np.asarray(direction)
                if np.any(point < low-1e-4) or np.any(point > high+1e-4):
                    raise AssertionError("Reported original-wall ray hit lies outside its authored bounds")
                hit_count += 1
    if hit_count != 192:
        raise AssertionError("All three snapshots must measure 64 rays on the original wall")
    video_path = args.visuals / "office-proximity.mp4"
    if hashlib.sha256(video_path.read_bytes()).hexdigest() != render["video_sha256"]:
        raise AssertionError("Video hash differs from render evidence")
    with imageio.get_reader(video_path) as reader:
        count = 0
        for frame in reader:
            if frame.shape != (1080, 1920, 3) or np.ptp(frame) == 0:
                raise AssertionError("Decoded video frame is invalid")
            count += 1
    if count != render["encoded_frames"] or len(frames) != render["source_observer_frames"]:
        raise AssertionError("Recorded or decoded video frame counts differ")
    for name in ("tof-heatmaps", "trajectory", "distance-speed"):
        with Image.open(args.visuals / f"{name}.png") as image:
            if min(image.size) < 700 or np.ptp(np.asarray(image)) == 0:
                raise AssertionError("Visualization image is invalid")
    output = {"accepted": True, "control_steps": total, "physics_calls": total*4,
              "original_wall_ray_hits_checked": hit_count, "decoded_video_frames": count,
              "source_observer_frames": len(frames), "measured_stop_samples": result["finish"]["stop_confirmation"]["stopped_samples"]}
    destination = args.visuals / "audit.json"
    if destination.exists():
        raise ValueError("Artifact audit output must be new")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
