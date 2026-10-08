import argparse
from bisect import bisect_right
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1920
HEIGHT = 1080
BACKGROUND = "#101b25"
PANEL = "#192a38"
PANEL_DARK = "#0b131c"
BORDER = "#365363"
TEXT = "#eff7f7"
MUTED = "#a9c0c6"
ACCENT = "#74dacb"
SUCCESS = "#74dacb"
FAILURE = "#ff9b8d"
FONT_DEFAULT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "abandoned", "unknown"}
TERMINAL_EVENT_TYPES = {"run.succeeded", "run.failed", "run.cancelled", "run.interrupted",
                        "run.abandoned"}
OBSERVER_CAMERA = "observer_follow.png"


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(export: Path, relative: str) -> Path:
    path = (export / relative).resolve()
    if not path.is_relative_to(export) or not path.is_file():
        raise ValueError(f"Recorded image is unavailable inside the export: {relative}")
    return path


def read_export(export: Path) -> tuple[dict, list[dict], dict, dict[str, list[dict]]]:
    run = load_json(export / "source" / "run.json")
    events = load_json(export / "source" / "events.json")
    rows = load_json(export / "frames.json")
    manifest = load_json(export / "manifest.json")
    if (not isinstance(events, list) or not events or not isinstance(rows, list)
            or run["id"] != manifest["runId"] or run["state"] != manifest["runState"]
            or run["state"] not in TERMINAL_STATES or len(events) != manifest["eventCount"]):
        raise ValueError("Run export identity, count, or terminal state is invalid")
    times = [timestamp(event["at"]) for event in events]
    if times != sorted(times) or any(event["sequence"] != index + 1
                                     for index, event in enumerate(events)):
        raise ValueError("Recorded event sequence or wall time is invalid")
    cameras: dict[str, list[dict]] = {}
    for row in rows:
        if row["kind"] not in ("simulation.frame", "agent.observation"):
            continue
        if row["file"] is None:
            if row["image"]["name"] == OBSERVER_CAMERA:
                raise ValueError("A recorded observer frame is unavailable")
            continue
        path = source_path(export, row["file"])
        if (path.stat().st_size != row["image"]["bytes"] or
                f"sha256:{file_sha256(path)}" != row["image"]["attachmentId"]):
            raise ValueError("Recorded camera bytes differ from the native attachment")
        frame = {"file": path, "wall": timestamp(row["eventAt"]),
                 "at": row["eventAt"], "event_sequence": row["eventSequence"],
                 "simulation_time_s": row["simulationTimeS"],
                 "name": row.get("cameraName", row["image"]["name"]),
                 "perception_source": row.get("perceptionSource"),
                 "sample_sequence": row.get("sampleSequence")}
        if row["kind"] == "simulation.frame" and (
                not isinstance(frame["simulation_time_s"], (int, float)) or
                not math.isfinite(frame["simulation_time_s"])):
            raise ValueError("Recorded simulation frame has no simulator time")
        cameras.setdefault(frame["name"], []).append(frame)
    if OBSERVER_CAMERA not in cameras or not cameras[OBSERVER_CAMERA]:
        raise ValueError("The run export contains no observer_follow frames")
    for name, frames in cameras.items():
        frames.sort(key=lambda item: item["event_sequence"])
        if any(frames[index]["wall"] > frames[index + 1]["wall"] or
               (frames[index]["simulation_time_s"] is not None and
                frames[index + 1]["simulation_time_s"] is not None and
                frames[index]["simulation_time_s"] > frames[index + 1]["simulation_time_s"])
               for index in range(len(frames) - 1)):
            raise ValueError(f"Recorded camera time is not monotonic: {name}")
    with Image.open(cameras[OBSERVER_CAMERA][0]["file"]) as first:
        if first.format != "PNG" or first.size not in ((640, 480), (1280, 720)):
            raise ValueError("The native observer camera dimensions are unsupported")
    return run, events, manifest, cameras


def holds(events: list[dict]) -> list[tuple[float, float, dict]]:
    durations = {"agent.output": 2.0, "plan.updated": 1.5,
                 "verification.completed": 3.0,
                 "run.succeeded": 3.5, "run.failed": 3.5,
                 "run.cancelled": 3.5, "run.interrupted": 3.5,
                 "run.abandoned": 3.5}
    return [(timestamp(event["at"]), durations[event["type"]], event)
            for event in events if event["type"] in durations and
            (event["type"] != "agent.output" or model_text(event).strip())]


def schedule(events: list[dict], observer_frames: list[dict],
             wall_speed: float) -> tuple[list[tuple], float, dict]:
    start = timestamp(events[0]["at"])
    end = timestamp(events[-1]["at"])
    frame_walls = [frame["wall"] for frame in observer_frames]
    frame_simulation = [frame["simulation_time_s"] for frame in observer_frames]
    if (not frame_walls or frame_walls[0] < start or frame_walls[-1] > end or
            any(frame_walls[index] == frame_walls[index + 1] and
                frame_simulation[index] != frame_simulation[index + 1]
                for index in range(len(frame_walls) - 1))):
        raise ValueError("Observer frame timestamps cannot define a monotonic schedule")

    def simulation_at(wall: float) -> float:
        if wall <= frame_walls[0]:
            return frame_simulation[0]
        if wall >= frame_walls[-1]:
            return frame_simulation[-1]
        index = bisect_right(frame_walls, wall) - 1
        left, right = frame_walls[index], frame_walls[index + 1]
        if left == right:
            raise ValueError("Observer simulation time has an ambiguous wall timestamp")
        fraction = (wall - left) / (right - left)
        return frame_simulation[index] + fraction * (frame_simulation[index + 1] - frame_simulation[index])

    read_holds = holds(events)
    nodes = sorted({start, end, *frame_walls, *(at for at, _, _ in read_holds)})
    hold_by_wall: dict[float, list[tuple[float, dict]]] = {}
    for at, duration, event in read_holds:
        hold_by_wall.setdefault(at, []).append((duration, event))
    playback = 0.0
    segments = []
    motion_floor_s = 0.0
    node_arrival_s = {}
    for index, wall in enumerate(nodes):
        if index:
            previous = nodes[index - 1]
            simulation_delta = simulation_at(wall) - simulation_at(previous)
            if simulation_delta < -1e-9:
                raise ValueError("Observer simulation time moved backward")
            interval = max((wall - previous) / wall_speed, max(0.0, simulation_delta))
            motion_floor_s += max(0.0, simulation_delta)
            if interval <= 0:
                raise ValueError("Recorded schedule has no positive wall interval")
            segments.append((playback, playback + interval, previous, wall, None))
            playback += interval
        node_arrival_s[wall] = playback
        for duration, event in hold_by_wall.get(wall, []):
            segments.append((playback, playback + duration, wall, wall, event))
            playback += duration
    if not segments:
        segments.append((0.0, 3.0, start, start, None))
        playback = 3.0
    if not math.isclose(motion_floor_s, frame_simulation[-1] - frame_simulation[0], abs_tol=1e-6):
        raise ValueError("Scheduled motion duration differs from recorded observer time")
    playback_ratios = []
    for index in range(len(frame_walls) - 1):
        simulation_delta = frame_simulation[index + 1] - frame_simulation[index]
        playback_delta = node_arrival_s[frame_walls[index + 1]] - node_arrival_s[frame_walls[index]]
        if playback_delta + 1e-9 < simulation_delta:
            raise ValueError("An observer interval plays faster than recorded simulation time")
        if simulation_delta > 0:
            playback_ratios.append(playback_delta / simulation_delta)
    return segments, playback, {
        "timeSchedule": "merged_recorded_observer_and_read_hold_nodes",
        "waitingWallPlaybackSpeed": wall_speed,
        "observerSimulationTimeRangeS": [frame_simulation[0], frame_simulation[-1]],
        "observerSimulationDurationS": frame_simulation[-1] - frame_simulation[0],
        "scheduledSimulationTimeFloorS": motion_floor_s,
        "observerScheduleNodes": len(frame_walls), "totalScheduleNodes": len(nodes),
        "motionIntervalRule": "max(wall_delta/waiting_wall_speed, adjacent_observer_simulation_delta)",
        "observerIntervalPlaybackFloorPassed": True,
        "minimumObserverIntervalPlaybackRatio": round(min(playback_ratios), 9) if playback_ratios else None,
    }


def wall_at(segments: list[tuple], second: float) -> tuple[float, dict | None]:
    for start, stop, wall_start, wall_stop, event in segments:
        if second < stop:
            fraction = (second - start) / (stop - start)
            return wall_start + (wall_stop - wall_start) * fraction, event
    return segments[-1][3], segments[-1][4]


def playback_at_wall(segments: list[tuple], wall: float) -> float:
    for start, stop, wall_start, wall_stop, event in segments:
        if event is None and wall_start <= wall <= wall_stop:
            fraction = (wall - wall_start) / (wall_stop - wall_start) if wall_stop > wall_start else 0.0
            return start + (stop - start) * fraction
    raise ValueError("Recorded wall time is absent from the playback schedule")


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
              font: ImageFont.FreeTypeFont, fill: str, right: int, bottom: int) -> None:
    bounds = draw.textbbox(xy, value, font=font)
    if bounds[2] > right or bounds[3] > bottom:
        raise ValueError(f"Video text crossed a panel boundary: {value!r}, {bounds}, {(right, bottom)}")
    draw.text(xy, value, font=font, fill=fill)


def wrapped(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.FreeTypeFont,
            width: int) -> list[str]:
    lines = []
    for paragraph in value.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else current + " " + word
            if draw.textlength(candidate, font=font) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if draw.textlength(word, font=font) <= width:
                current = word
                continue
            for character in word:
                if current and draw.textlength(current + character, font=font) > width:
                    lines.append(current)
                    current = character
                else:
                    current += character
        lines.append(current)
    return lines


def draw_block(draw: ImageDraw.ImageDraw, value: str, xy: tuple[int, int],
               font: ImageFont.FreeTypeFont, width: int, line_height: int,
               maximum: int, right: int, bottom: int, fill: str = TEXT) -> None:
    lines = wrapped(draw, value, font, width)
    if len(lines) > maximum:
        lines = lines[:maximum]
        final = lines[-1]
        while final and draw.textlength(final + "…", font=font) > width:
            final = final[:-1]
        lines[-1] = final + "…"
    for index, line in enumerate(lines):
        draw_text(draw, (xy[0], xy[1] + index * line_height), line,
                  font, fill, right, bottom)


def panel(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int],
          title: str, font: ImageFont.FreeTypeFont) -> None:
    draw.rounded_rectangle(bounds, radius=14, fill=PANEL, outline=BORDER, width=2)
    draw_text(draw, (bounds[0] + 18, bounds[1] + 12), title, font, ACCENT,
              bounds[2] - 18, bounds[1] + 45)


def latest(events: list[dict], times: list[float], wall: float, event_type: str,
           member: str | None = None) -> dict | None:
    for event in reversed(events[:bisect_right(times, wall)]):
        if event["type"] == event_type and (member is None or
           member.lower() in str(event["detail"].get("member", "")).lower()):
            return event
    return None


def latest_tool_result(events: list[dict], times: list[float], wall: float,
                       tool_name: str | tuple[str, ...]) -> dict | None:
    names = (tool_name,) if isinstance(tool_name, str) else tool_name
    for event in reversed(events[:bisect_right(times, wall)]):
        if event["type"] == "tool.completed" and event["detail"].get("tool") in names:
            result = event["detail"].get("result")
            if not isinstance(result, dict):
                raise ValueError(f"Recorded {tool_name} result is not an object")
            return result
    return None


def progress_motion_evidence(result: dict) -> tuple[str, str, str]:
    guard = result.get("motion_guard")
    contact = result.get("contact_evidence")
    reason = guard.get("reason", "未记录") if isinstance(guard, dict) else "未记录"
    distance = guard.get("central_tof_closest_mm", "未记录") if isinstance(guard, dict) else "未记录"
    samples = (contact.get("non_ground_external_contact_samples_total", "未记录")
               if isinstance(contact, dict) else "未记录")
    return str(reason), "无有效读数" if distance is None else str(distance), str(samples)


def model_text(event: dict | None) -> str:
    if event is None:
        return ""
    message = event["detail"].get("message", {})
    blocks = message.get("content", []) if isinstance(message, dict) else []
    return "\n".join(block.get("text", "") for block in blocks
                     if isinstance(block, dict) and block.get("type") == "text" and block.get("text"))


def selected(value: dict, names: tuple[str, ...]) -> dict:
    return {name: value[name] for name in names if name in value}


def tool_summary(event: dict) -> str:
    detail = event["detail"]
    tool = detail.get("tool", "tool")
    result = detail.get("result", {})
    if not isinstance(result, dict):
        return f"{tool} · {json.dumps(result, ensure_ascii=False)}"
    if tool == "microduck.scene_info":
        visible = selected(result, ("scene", "frame", "rooms", "scene_id", "frame_id", "solver"))
    elif tool == "microduck.policy_catalog":
        visible = {"revision": result.get("revision"),
                   "policies": [policy["name"] for policy in result.get("policies", [])]}
    elif tool == "microduck.read_sensor":
        measurements = result.get("measurements", {})
        visible = selected(result, ("sensor", "sequence"))
        if result.get("sensor") == "tof":
            valid = [distance for distance, status in zip(measurements["tof_distance_mm"],
                                                            measurements["tof_status"], strict=True)
                     if status == 5]
            visible.update({"valid_hits": len(valid), "no_hit": len(measurements["tof_status"]) - len(valid),
                            "nearest_valid_mm": min(valid) if valid else None})
        elif result.get("sensor") == "head_rgb":
            visible.update(selected(measurements, ("rgb_width", "rgb_height")))
        elif result.get("sensor") == "odometry":
            visible["odometry"] = selected(measurements.get("odometry", {}), ("x_m", "y_m", "yaw_rad"))
        elif result.get("sensor") == "imu":
            visible.update(selected(measurements, ("angular_velocity_rad_s", "projected_gravity")))
        elif result.get("sensor") == "joint_state":
            visible["joint_count"] = len(measurements["joint_names"])
    elif tool == "microduck.inspect_scene":
        targets = [f"{target['label']} · {target['distance_m']:.2f} m · bearing {target['bearing_deg']:+.1f}°"
                   if target["distance_status"] == "valid" else
                   f"{target['label']} · {target['distance_status']}" for target in result.get("targets", [])[:3]]
        return (f"{tool} · {result['prompt']} · captured seq {result['sequence']}\n"
                f"detection: {result['detection_source']} · distance: {result['distance_source']}\n" +
                "\n".join(targets or ["No visible target"]))
    elif tool == "microduck.set_command":
        visible = selected(result, ("effective_after_sequence", "command"))
    elif tool in ("microduck.walk", "microduck.rotate"):
        visible = selected(result, ("prepared", "arguments", "policy_name", "next_action"))
    elif tool in ("microduck.task_progress", "microduck.observe", "microduck.wait_for_motion"):
        reason, distance, samples = progress_motion_evidence(result)
        position = result.get("body_position_m")
        coordinates = ("未记录" if position is None else
                       "(" + ",".join(f"{value:.3f}" for value in position) + ")m")
        execution = result.get("execution")
        state = execution.get("state", "未记录") if isinstance(execution, dict) else "未记录"
        text = (f"{tool} · seq={result.get('sequence', '未记录')} · {coordinates} · {state}\n"
                f"guard={reason} · central ToF={distance} mm · non-ground samples={samples}")
        if motion := result.get("metric_motion"):
            text += (f"\n{motion['operation']} {motion['measured']:.3f}/{motion['requested']:.3f} {motion['unit']}"
                     f" · error {motion['error']:.3f} · {motion['phase']}")
        if isinstance(execution, dict) and "remaining_wall_time_s" in execution:
            text += f"\nremaining {execution['remaining_wall_time_s']:.1f} s / {execution['remaining_actions']} controls"
        if perception := result.get("perception"):
            text += f"\n{perception['prompt']} · {perception['distance_source']} · {len(perception['targets'])} targets"
        return text
    elif tool == "microduck.select_policy":
        visible = selected(result, ("policy_name", "kind", "encoding", "duration_s"))
        visible["action_spec_version"] = result.get("action_spec", {}).get("version")
    elif tool == "microduck.finish_policy":
        visible = selected(result, ("accepted",))
        visible["stop_confirmation"] = selected(result["stop_confirmation"],
                                                ("zero_control_steps", "stopped_samples"))
    elif tool.startswith("execution."):
        visible = selected(result.get("execution", {}),
                           ("state", "control_steps", "raw_sim_steps", "device_confirmed",
                            "stop_reason", "generation"))
    elif tool == "planning.read":
        visible = selected(result, ("activeGoalId", "attemptId", "finalGoalId"))
        visible["plan_version"] = (result.get("plan") or {}).get("version")
    elif tool == "planning.update":
        visible = {"plan_version": result.get("plan", {}).get("version"),
                   "items": [selected(item, ("description", "status"))
                             for item in result.get("plan", {}).get("items", [])]}
    else:
        visible = selected(result, ("state", "generation", "device_confirmed", "policy_name",
                                    "command", "sequence", "simulation_time_s", "body_position_m",
                                    "check_id", "satisfied", "evidence", "accepted", "error"))
        if "checks" in result:
            visible["checks"] = result["checks"]
    return f"{tool} · {json.dumps(visible, ensure_ascii=False, separators=(',', ':'))}"


def native_check(event: dict) -> dict:
    detail = event["detail"]
    facts = detail["facts"] if event["type"] == "verification.checked" else detail["result"]["checks"]
    fact = next(item for item in facts if item["check_id"] == "goal_reached")
    reason = json.loads(fact["reason"])
    evidence = reason["evidence"]
    return {"check_id": fact["check_id"], "value": fact["value"],
            "position_m": evidence["robot_world_position_m"],
            "held_ticks": evidence["held_ticks"],
            "required_hold_ticks": evidence["required_hold_ticks"],
            "target": evidence["target"]}


def native_target_text(target: dict) -> str:
    if "room" in target:
        return target["room"]
    elif "target_xy_m" in target:
        return f"distance={target['distance_xy_m']:.3f}/{target['threshold_m']:.3f} m"
    raise ValueError("Native verification target is unsupported")


def native_check_text(event: dict) -> str:
    check = native_check(event)
    position = check["position_m"]
    destination = native_target_text(check["target"])
    return (f"{check['check_id']}={str(check['value']).lower()} · "
            f"{destination} · position=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f}) m · "
            f"hold={check['held_ticks']}/{check['required_hold_ticks']}")


def event_summary(event: dict) -> str:
    detail = event["detail"]
    kind = event["type"]
    if kind == "tool.failed":
        return f"{detail['tool']} · {detail['error']}"
    if kind == "agent.output":
        return model_text(event)
    if kind == "plan.updated":
        items = detail.get("plan", {}).get("items", [])
        return " | ".join(f"{item.get('status', '')}: {item.get('description', '')}" for item in items)
    if kind.startswith("tool."):
        return tool_summary(event)
    if kind == "verification.checked":
        return native_check_text(event)
    if kind == "verification.completed":
        return f"{detail['result']['status']} · {native_check_text(event)}"
    if kind.startswith("run."):
        return json.dumps(selected(detail, ("state", "status", "reason", "error", "source", "scenario")),
                          ensure_ascii=False, separators=(",", ":"))
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))


def active_frame(frames: list[dict], wall: float) -> dict | None:
    times = [frame["wall"] for frame in frames]
    index = bisect_right(times, wall) - 1
    return frames[index] if index >= 0 else None


def load_camera_image(frame: dict | None, cache: dict, size: tuple[int, int]) -> Image.Image | None:
    if frame is None:
        return None
    key = (frame["file"], size)
    if cache.get("key") != key:
        with Image.open(frame["file"]) as image:
            if image.format != "PNG":
                raise ValueError("A recorded camera file is not PNG")
            fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
            background = Image.new("RGB", size, PANEL_DARK)
            background.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
            cache["image"] = background
        cache["key"] = key
    return cache["image"]


def render_frame(run: dict, events: list[dict], times: list[float], cameras: dict[str, list[dict]],
                 wall: float, hold: dict | None, wall_speed: float, logo: Image.Image,
                 fonts: dict, caches: dict) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    canvas.paste(logo, (34, 24), logo)
    draw_text(draw, (130, 24), "OH MY DUCK  ·  AGENTIC MICRODUCK",
              fonts["title"], TEXT, 1190, 72)
    environment = run["configuration"]["launchProfile"]["environment"]
    label = ("Isaac Lab / Newton / BAM" if environment.startswith("Isaac Lab Newton/BAM:")
             else "CPU MuJoCo / BAM · official apartment")
    draw_text(draw, (132, 80), f"{label} · waiting {wall_speed:g}× · motion ≤1×",
              fonts["small"], MUTED, 1350, 112)
    brain = run["configuration"]["models"]["brain"]
    draw_text(draw, (1195, 80), f"BRAIN · {brain['model']} · {brain.get('reasoningEffort', 'provider default')}",
              fonts["caption"], MUTED, 1638, 112)
    visible_events = events[:bisect_right(times, wall)]
    terminal_event = next((event for event in reversed(visible_events)
                           if event["type"] in TERMINAL_EVENT_TYPES), None)
    final_outcome_visible = terminal_event is not None or wall >= times[-1]
    status_color = (SUCCESS if run["state"] == "succeeded" else FAILURE) if final_outcome_visible else MUTED
    draw.rounded_rectangle((1660, 30, 1888, 102), radius=12, fill=PANEL, outline=status_color, width=2)
    if final_outcome_visible:
        draw_text(draw, (1674, 37), "FINAL OUTCOME", fonts["caption"], status_color, 1874, 64)
        draw_text(draw, (1674, 66), run["state"].upper(), fonts["body"], status_color, 1874, 96)
    else:
        draw_text(draw, (1674, 54), "RUN IN PROGRESS", fonts["small"], status_color, 1874, 90)

    draw.rounded_rectangle((30, 142, 1194, 1018), radius=15, fill=PANEL_DARK, outline=BORDER, width=2)
    observer = active_frame(cameras[OBSERVER_CAMERA], wall)
    main = load_camera_image(observer, caches["observer"], (1152, 864))
    if main is not None:
        canvas.paste(main, (36, 148))
        draw.rounded_rectangle((48, 957, 742, 1002), radius=9, fill=PANEL_DARK)
        draw_text(draw, (62, 968),
                  f"OBSERVER FOLLOW  ·  event #{observer['event_sequence']}  ·  sim {observer['simulation_time_s']:.2f} s",
                  fonts["caption"], TEXT, 728, 994)
    else:
        draw_text(draw, (330, 540), "Waiting for recorded observer frame",
                  fonts["body"], MUTED, 1110, 590)
    progress = latest_tool_result(events, times, wall,
                                  ("microduck.task_progress", "microduck.observe", "microduck.wait_for_motion"))
    if progress is not None:
        reason, distance, samples = progress_motion_evidence(progress)
        guard = progress.get("motion_guard")
        guard_sequence = guard.get("sequence", "未记录") if isinstance(guard, dict) else "未记录"
        draw.rounded_rectangle((48, 160, 676, 238), radius=9, fill=PANEL_DARK)
        draw_text(draw, (62, 170),
                  f"LATEST TASK PROGRESS seq={progress.get('sequence', '未记录')}  ·  guard {reason}",
                  fonts["caption"], TEXT, 660, 199)
        draw_text(draw, (62, 203),
                  f"guard seq={guard_sequence}  ·  ToF {distance} mm  ·  non-ground samples {samples}",
                  fonts["caption"], TEXT, 660, 231)
        scene = latest_tool_result(events, times, wall, "microduck.scene_info")
        goal = scene.get("goal") if scene is not None else None
        position = progress.get("body_position_m")
        if goal is not None and goal["kind"] == "point" and position is not None:
            target = goal["target_xy_m"]
            error = math.dist(position[:2], target)
            draw.rounded_rectangle((48, 908, 902, 950), radius=9, fill=PANEL_DARK)
            draw_text(draw, (62, 918),
                      f"WORLD XY ({position[0]:.3f}, {position[1]:.3f}) m  ·  GOAL DISTANCE {error:.3f} / {goal['distance_m']:.3f} m",
                      fonts["caption"], ACCENT, 888, 946)
    head_frames = cameras.get("head_camera.png", cameras.get("head_rgb.png"))
    if head_frames:
        head = active_frame(head_frames, wall)
        inset = load_camera_image(head, caches["head"], (256, 192))
        if inset is not None:
            draw.rectangle((908, 792, 1180, 1002), fill=PANEL_DARK, outline=ACCENT, width=2)
            canvas.paste(inset, (916, 800))
            caption = ("HEAD CAPTURE · SIM GT" if head.get("perception_source") == "simulator_ground_truth"
                       else "HEAD CAPTURE · RGB")
            draw.rectangle((916, 800, 1172, 823), fill=PANEL_DARK)
            draw_text(draw, (925, 801), f"seq {head['sample_sequence']} · event #{head['event_sequence']}",
                      fonts["caption"], TEXT, 1168, 822)
            draw_text(draw, (928, 964), caption, fonts["caption"],
                      TEXT, 1174, 996)

    planner = latest(events, times, wall, "agent.output", "planner")
    if planner and model_text(planner):
        panel(draw, (1210, 142, 1888, 314), "PLANNER TEXT · RECORDED", fonts["section"])
        draw_text(draw, (1230, 190), f"Event #{planner['sequence']}  ·  {planner['at']}",
                  fonts["caption"], MUTED, 1870, 220)
        draw_block(draw, model_text(planner), (1230, 224), fonts["small"], 630, 25,
                   3, 1870, 307)
    else:
        panel(draw, (1210, 142, 1888, 314), "RECORDED TASK · RUN INSTRUCTION", fonts["section"])
        draw_block(draw, run["instruction"], (1230, 190), fonts["small"], 630, 25,
                   4, 1870, 307)

    panel(draw, (1210, 324, 1888, 456), "PLAN AND EXECUTION", fonts["section"])
    plan_event = latest(events, times, wall, "plan.updated")
    plan_text = event_summary(plan_event) if plan_event else "Plan pending"
    draw_block(draw, plan_text, (1230, 368), fonts["small"], 635, 25, 2, 1870, 420)
    execution = latest(events, times, wall, "execution.updated")
    if execution:
        state = execution["detail"].get("execution", {})
        value = (f"{state.get('state', 'unknown')} · {state.get('control_steps', 0)} controls / "
                 f"{state.get('raw_sim_steps', 0)} physics")
        if state.get("stop_reason"):
            value += f" · {state['stop_reason']}"
        draw_text(draw, (1230, 427), value, fonts["caption"], MUTED, 1870, 451)

    panel(draw, (1210, 466, 1888, 866), "TIME-ORDERED AGENTIC TRACE", fonts["section"])
    current = events[:bisect_right(times, wall)]
    trace = [event for event in current if (event["type"] == "agent.output" and model_text(event).strip()) or
             event["type"] == "plan.updated" or event["type"] in {"tool.completed", "tool.failed"} or
             event["type"].startswith("verification.") or event["type"].startswith("run.")]
    trace = trace[-4:]
    for index, event in enumerate(trace):
        y = 510 + index * 88
        draw.rounded_rectangle((1224, y, 1874, y + 82), radius=8, fill=PANEL_DARK)
        label = f"#{event['sequence']}  {event['type']}  ·  {event['at'][11:19]}"
        draw_text(draw, (1236, y + 7), label, fonts["caption"], ACCENT, 1860, y + 32)
        draw_block(draw, event_summary(event), (1236, y + 35), fonts["caption"], 622, 21,
                   2, 1860, y + 79, fill=TEXT)

    panel(draw, (1210, 876, 1888, 1018), "FORMAL VERDICT · NATIVE GT", fonts["section"])
    check = latest(events, times, wall, "verification.checked")
    verdict = latest(events, times, wall, "verification.completed")
    if check:
        fact = native_check(check)
        position = fact["position_m"]
        draw_text(draw, (1230, 923),
                  f"{fact['check_id']}: {str(fact['value']).lower()} · {native_target_text(fact['target'])}",
                  fonts["small"], TEXT, 1870, 950)
        draw_text(draw, (1230, 949),
                  f"position (m): {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f} · "
                  f"hold {fact['held_ticks']}/{fact['required_hold_ticks']}",
                  fonts["caption"], TEXT, 1870, 974)
    if verdict:
        status = verdict["detail"].get("result", {}).get("status", "recorded")
        draw_text(draw, (1230, 978), f"Formal result: {status}", fonts["body"],
                  SUCCESS if status == "passed" else FAILURE, 1870, 1010)
    elif not check:
        draw_text(draw, (1230, 934), "Native check pending", fonts["body"], MUTED, 1870, 972)

    start, end = times[0], times[-1]
    fraction = 1.0 if end == start else (wall - start) / (end - start)
    draw.rounded_rectangle((30, 1030, 1888, 1064), radius=8, fill=PANEL, outline=BORDER)
    draw.rounded_rectangle((33, 1033, 33 + max(1, int(1851 * fraction)), 1061),
                           radius=6, fill="#24666a")
    label = f"RECORDED WALL +{wall - start:.1f} s / {end - start:.1f} s"
    draw_text(draw, (48, 1036), label, fonts["caption"], TEXT, 750, 1060)
    if hold:
        draw_text(draw, (1000, 1036), f"READ HOLD  ·  #{hold['sequence']} {hold['type']}",
                  fonts["caption"], TEXT, 1870, 1060)
    return canvas


def encode_video(export: Path, output: Path, review_dir: Path, font_path: Path, fps: int,
                 wall_speed: float) -> dict:
    if output.exists() or fps <= 0 or wall_speed <= 0 or not font_path.is_file():
        raise ValueError("Output must be new; FPS, wall speed, and font must be valid")
    run, events, manifest, cameras = read_export(export)
    source_event_count = len(events)
    source_observer_count = len(cameras[OBSERVER_CAMERA])
    terminal_index = next((index for index, event in enumerate(events)
                           if event["type"] in TERMINAL_EVENT_TYPES), None)
    if terminal_index is None:
        raise ValueError("The recorded run has no terminal event")
    events = events[:terminal_index + 1]
    terminal_event = events[-1]
    cameras = {name: [frame for frame in frames
                      if frame["event_sequence"] <= terminal_event["sequence"]]
               for name, frames in cameras.items()}
    if not cameras[OBSERVER_CAMERA]:
        raise ValueError("The run has no observer frame before its terminal event")
    segments, duration, schedule_details = schedule(events, cameras[OBSERVER_CAMERA], wall_speed)
    frame_count = max(1, math.ceil(duration * fps) + 1)
    fonts = {name: ImageFont.truetype(str(font_path), size) for name, size in
             {"title": 34, "section": 24, "body": 22, "small": 19, "caption": 16}.items()}
    logo_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "oh-my-duck.png"
    with Image.open(logo_path) as original:
        logo = original.convert("RGBA").resize((82, 82), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".ffmpeg.log")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    times = [timestamp(event["at"]) for event in events]
    caches = {"observer": {}, "head": {}}
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        if process.stdin is None:
            raise RuntimeError("FFmpeg video input is unavailable")
        try:
            for index in range(frame_count):
                wall, hold = wall_at(segments, min(duration, index / fps))
                frame = render_frame(run, events, times, cameras, wall, hold, wall_speed,
                                     logo, fonts, caches)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg could not encode the recorded video: {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.wait()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height,nb_frames,duration",
                            "-show_entries", "format=duration", "-of", "json", str(output)],
                           check=True, capture_output=True, text=True)
    probe_data = json.loads(probe.stdout)
    stream = probe_data["streams"][0]
    if (stream["codec_name"] != "h264" or stream["width"] != WIDTH or
            stream["height"] != HEIGHT or int(stream["nb_frames"]) != frame_count):
        raise ValueError("Encoded video dimensions, codec, or frame count differ")
    observer_frames = cameras[OBSERVER_CAMERA]
    verifier_hold = next((segment for segment in segments
                          if segment[4] is not None and segment[4]["type"] == "verification.completed"), None)
    review_times = {
        "start": 0.0,
        "motion_early": playback_at_wall(segments, observer_frames[len(observer_frames) // 3]["wall"]),
        "motion_late": playback_at_wall(segments, observer_frames[2 * len(observer_frames) // 3]["wall"]),
        "verifier": ((verifier_hold[0] + verifier_hold[1]) / 2 if verifier_hold is not None
                     else duration * 0.75),
        "final": max(0.0, duration - 1 / fps),
    }
    keyframes = []
    for name, second in review_times.items():
        destination = review_dir / f"{output.stem}-{name}.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss",
                        f"{second:.3f}", "-i", str(output), "-frames:v", "1", "-y",
                        str(destination)], check=True)
        with Image.open(destination) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError("Extracted review frame has the wrong dimensions")
        keyframes.append(str(destination))
    environment = run["configuration"]["launchProfile"]["environment"]
    backend_file = "isaac_official.py" if environment.startswith("Isaac Lab Newton/BAM:") else "simulation.py"
    backend_source = Path(__file__).resolve().parents[1] / "src" / "oh_my_duck" / "robotics" / "backends" / backend_file
    report = {"runId": run["id"], "runState": run["state"], "runError": run.get("error"),
              "sourceEventCount": source_event_count, "renderedEventCount": len(events),
              "terminalEventSequence": terminal_event["sequence"],
              "terminalEventType": terminal_event["type"], "terminalEventAt": terminal_event["at"],
              "sourceObserverFrameCount": source_observer_count,
              "environment": environment,
              "renderedObserverFrameCount": len(cameras[OBSERVER_CAMERA]),
              "optionalCameraNames": sorted(set(cameras) - {OBSERVER_CAMERA}),
              "width": WIDTH, "height": HEIGHT, "fps": fps, "frameCount": frame_count,
              "renderedWallDurationS": times[-1] - times[0], "wallPlaybackSpeed": wall_speed,
              "readHoldCount": len(holds(events)), "expectedDurationS": duration,
              "encodedDurationS": float(probe_data["format"]["duration"]),
              "textBoundaryChecks": "passed for every encoded frame",
              "formalVerdict": (latest(events, times, times[-1], "verification.completed") or
                                {}).get("detail", {}).get("result", {}).get("status"),
              "recordedReasoningEvents": sum(event["type"] == "agent.output" and
                                             any(block.get("type") == "reasoning" for block in
                                                 event["detail"].get("message", {}).get("content", []))
                                             for event in events),
              "keyframes": keyframes, "sourceManifest": str(export / "manifest.json"),
              "keyframePlaybackTimesS": review_times,
              "videoSha256": file_sha256(output),
              "sourceManifestSha256": file_sha256(export / "manifest.json"),
              "sourceRunSha256": file_sha256(export / "source" / "run.json"),
              "sourceEventsSha256": file_sha256(export / "source" / "events.json"),
              "observerBackendSourceSha256": file_sha256(backend_source),
              "rendererSourceSha256": file_sha256(Path(__file__)),
              **schedule_details}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=FONT_DEFAULT)
    parser.add_argument("--review-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".cache" / "demo-review")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--wall-speed", type=float, default=4.0)
    args = parser.parse_args()
    report = encode_video(args.export.resolve(), args.output.resolve(), args.review_dir.resolve(), args.font,
                          args.fps, args.wall_speed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
