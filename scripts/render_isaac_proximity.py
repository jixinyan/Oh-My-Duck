import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {"warmup": "#8198ae", "approach": "#48d5be", "stop": "#ffbe65"}


def load(root, name):
    return json.loads((root / f"{name}.json").read_text())


def central(sample):
    values = np.asarray(sample["tof_distance_mm"]).reshape(8, 8)
    valid = np.asarray(sample["tof_status"]).reshape(8, 8) == 5
    selected = values[2:6, 2:6][valid[2:6, 2:6]]
    return float(selected.min()) if selected.size else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result, samples, frames, tools = [load(args.input, name) for name in ("result", "samples", "frames", "tools")]
    if not result["accepted"] or not frames or not samples:
        raise ValueError("Accepted native evidence and camera frames are required")
    times = np.asarray([s["simulation_time_s"] for s in samples])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Native sample times must be strictly increasing")
    guard_time = result["guard"]["sequence"] / 50
    stop_time = next(s["simulation_time_s"] for s in samples if s["phase"] == "stop")
    positions = np.asarray([s["body_position_m"] for s in samples])
    twists = np.asarray([s["body_twist"] for s in samples])
    distances = np.asarray([central(s) for s in samples])
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#111c29", "axes.facecolor": "#182738",
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(times, distances, color="#48d5be", label="Central 4 x 4 minimum (status=5)")
    axes[0].axhline(90, color="#ff817b", linestyle="--", label="Proximity threshold: 90 mm")
    axes[0].set_ylabel("ToF distance / mm")
    axes[1].plot(times, np.linalg.norm(twists[:, :2], axis=1), color="#ffbe65", label="Planar speed")
    axes[1].axhline(0.025, color="#ff817b", linestyle="--", label="Stop threshold: 0.025 m/s")
    axes[1].set_ylabel("Speed / m/s")
    axes[2].plot(times, np.abs(twists[:, 2]), color="#88b8ff", label="Absolute yaw rate")
    axes[2].axhline(0.08, color="#ff817b", linestyle="--", label="Stop threshold: 0.08 rad/s")
    axes[2].set_ylabel("Yaw rate / rad/s")
    axes[2].set_xlabel("Recorded simulation time / s")
    for axis in axes:
        axis.axvline(guard_time, color="white", linestyle=":", label="Native proximity pause")
        axis.axvspan(stop_time, times[-1], color=COLORS["stop"], alpha=0.09)
        axis.legend(loc="upper right", fontsize=9)
        axis.grid(alpha=0.15)
    figure.suptitle("OH MY DUCK | Office ToF and measured stopping", fontsize=18)
    figure.savefig(args.output / "distance-speed.png", dpi=160)
    figure.savefig(args.output / "distance-speed.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    wall = result["configuration"]["expected_wall"]
    low, high = np.asarray(wall["min"]), np.asarray(wall["max"])
    axis.add_patch(Rectangle(low[:2], *(high-low)[:2], color="#88b8ff", alpha=0.65, label="Original Office wall"))
    for phase, color in COLORS.items():
        mask = np.asarray([s["phase"] == phase for s in samples])
        axis.plot(positions[mask, 0], positions[mask, 1], color=color, linewidth=2.5, label=phase)
    guard = np.asarray(result["guard"]["body_position_m"])
    axis.scatter(guard[0], guard[1], color="#ff817b", marker="X", s=110, label="Proximity pause")
    axis.scatter(positions[-1, 0], positions[-1, 1], color="white", marker="s", s=60, label="Confirmed physical rest")
    axis.set(xlim=(positions[:, 0].min()-.1, high[0]+.08),
             ylim=(positions[:, 1].min()-.12, positions[:, 1].max()+.12),
             xlabel="World x / m", ylabel="World y / m", title="Measured trajectory near the original wall")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.15)
    axis.legend(fontsize=9, loc="upper left")
    figure.savefig(args.output / "trajectory.png", dpi=160)
    figure.savefig(args.output / "trajectory.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for label, axis in zip(("initial", "guard", "final"), axes, strict=True):
        state = load(args.input, "sensor-" + label)
        measured = (next(s for s in samples if s["sequence"] == result["guard"]["sequence"])
                    if label == "guard" else state["measurements"])
        grid = np.asarray(measured["tof_distance_mm"]).reshape(8, 8).astype(float)
        grid[np.asarray(measured["tof_status"]).reshape(8, 8) != 5] = np.nan
        im = axis.imshow(grid, vmin=0, vmax=400, cmap="viridis")
        axis.add_patch(Rectangle((1.5, 1.5), 4, 4, fill=False, edgecolor="#ff817b", linewidth=2))
        origin = "trigger observation" if label == "guard" else "fresh snapshot"
        axis.set_title(f"{label.upper()} | control {state['sequence']}\n{origin}")
        axis.set_xticks(range(8)); axis.set_yticks(range(8))
        for row in range(8):
            for col in range(8):
                axis.text(col, row, "--" if np.isnan(grid[row, col]) else str(int(grid[row, col])),
                          ha="center", va="center", fontsize=9, color="white")
    figure.colorbar(im, ax=axes, label="ToF distance / mm", shrink=.65)
    figure.suptitle("8 x 8 ToF | central guard region outlined | invalid readings masked", fontsize=15)
    figure.savefig(args.output / "tof-heatmaps.png", dpi=160)
    figure.savefig(args.output / "tof-heatmaps.svg")
    plt.close(figure)

    font = ImageFont.truetype(str(args.font), 20)
    title_font = ImageFont.truetype(str(args.font), 28)
    frame_times = [f["simulation_time_s"] for f in frames]
    video_path = args.output / "office-proximity.mp4"
    count = int(np.ceil((times[-1]-frame_times[0])*10))+21
    with imageio.get_writer(video_path, fps=10, codec="libx264", quality=8, macro_block_size=None) as writer:
        for index in range(count):
            timestamp = min(float(times[-1]), frame_times[0] + index/10)
            frame_index = max(0, bisect_right(frame_times, timestamp+1e-8)-1)
            sample = samples[max(0, bisect_right(times, timestamp+1e-8)-1)]
            canvas = Image.new("RGB", (1920, 1080), "#111c29")
            draw = ImageDraw.Draw(canvas)
            draw.text((30, 24), "OH MY DUCK | Office proximity control", font=title_font, fill="white")
            draw.text((30, 70), f"Isaac Lab + Newton + BAM | alpha_walking | simulation {timestamp:.2f} s", font=font, fill="#aebfcf")
            with Image.open(args.input / frames[frame_index]["file"]) as image:
                if image.size != (1280, 720):
                    raise ValueError("Native observer resolution differs from 1280 x 720")
                canvas.paste(image.convert("RGB"), (24, 125))
            draw.text((1340, 125), "NATIVE EXECUTION TRACE", font=title_font, fill="#48d5be")
            visible = [t for t in tools if t["sequence"] <= sample["sequence"]][-5:]
            for number, tool in enumerate(visible):
                y = 175+number*58
                draw.text((1340, y), f"{tool['sequence']:04d}  {tool['operation']}", font=font, fill="white")
                draw.text((1340, y+25), tool["phase"], font=font, fill="#aebfcf")
            draw.text((1340, 495), f"PHASE: {sample['phase'].upper()}", font=title_font, fill=COLORS[sample["phase"]])
            grid = np.asarray(sample["tof_distance_mm"]).reshape(8, 8)
            valid = np.asarray(sample["tof_status"]).reshape(8, 8) == 5
            for row in range(8):
                for col in range(8):
                    color = tuple(int(c*255) for c in plt.get_cmap("viridis")(min(grid[row, col]/400, 1))[:3]) if valid[row, col] else "#293342"
                    x, y = 1350+col*58, 560+row*48
                    draw.rectangle((x, y, x+54, y+44), fill=color)
                    draw.text((x+5, y+9), str(grid[row, col]) if valid[row, col] else "--", font=font, fill="white")
            draw.rectangle((1464, 654, 1698, 850), outline="#ff817b", width=3)
            distance = central(sample)
            draw.text((30, 880), f"control {sample['sequence']} | central ToF {distance:.0f} mm | planar speed {np.linalg.norm(sample['body_twist'][:2]):.3f} m/s", font=title_font, fill="white")
            draw.text((30, 925), "PROXIMITY PAUSE CONFIRMED" if timestamp >= guard_time else "APPROACHING ORIGINAL OFFICE WALL", font=title_font, fill="#ffbe65")
            if timestamp >= times[-1]:
                confirmation = result["finish"]["stop_confirmation"]
                draw.text((1340, 950), f"PHYSICAL REST: {confirmation['stopped_samples']} samples", font=font, fill="#48d5be")
            draw.text((30, 980), "Recorded native execution; VLM and formal navigation verdict pending. Source observer cadence: 10 Hz.", font=font, fill="#aebfcf")
            writer.append_data(np.asarray(canvas))
    metadata = {"source_observer_frames": len(frames), "source_samples": len(samples), "encoded_frames": count,
                "fps": 10, "width": 1920, "height": 1080,
                "schedule": "Recorded simulation time; latest source frame held between captures; final state held for 2 seconds",
                "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256((args.input / "result.json").read_bytes()).hexdigest(),
                "guard_time_s": guard_time, "stop_command_time_s": stop_time}
    (args.output / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
