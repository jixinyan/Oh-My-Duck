import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--scene-config", type=Path, required=True)
    args = parser.parse_args()
    root = args.export.resolve(strict=True)
    events = json.loads((root / "source/events.json").read_text())
    run = json.loads((root / "source/run.json").read_text())
    frames = json.loads((root / "frames.json").read_text())
    config = json.loads(args.scene_config.read_text())
    if run["state"] != "succeeded":
        raise AssertionError("VLN acceptance requires native task success")
    tools = [event for event in events if event["type"] == "tool.completed"]
    model_calls = [event["detail"]["data"]["name"] for event in events if event["type"] == "dsh.tool-call"]
    for name in ("microduck__walk", "microduck__rotate"):
        if model_calls.count(name) < 2:
            raise AssertionError("VLN requires multiple actual model walk and turn calls")
    inspections = [event["detail"]["result"] for event in tools
                   if event["detail"]["tool"] == "microduck.inspect_scene"]
    visible = [result for result in inspections if any(target["label"] == "desk" and
               target["distance_status"] == "valid" and math.isfinite(target["distance_m"]) and
               target["distance_m"] > 0 for target in result["targets"])]
    if len({(result["episode_id"], result["sequence"]) for result in visible}) < 3:
        raise AssertionError("VLN lacks three distinct physical desk observations")
    for result in visible:
        if result["distance_source"] != "simulator_ground_truth":
            raise AssertionError("This benchmark requires explicit simulator-ground-truth distances")
        reference = result["image_ref"]
        row = next(row for row in frames if row["image"]["attachmentId"] == reference["attachmentId"])
        image = root / row["file"]
        if f"sha256:{hashlib.sha256(image.read_bytes()).hexdigest()}" != reference["attachmentId"]:
            raise AssertionError("Perception image differs from the model tool attachment")
        with Image.open(image) as captured:
            captured.verify()
    verification = [event for event in events if event["type"] == "verification.completed"]
    if len(verification) != 1 or verification[0]["detail"]["result"]["status"] != "passed":
        raise AssertionError("VLN lacks one passed independent Verifier result")
    check = verification[0]["detail"]["result"]["checks"][0]
    evidence = json.loads(check["reason"])["evidence"]
    target = evidence["target"]
    waypoints = config["goal"]["waypoints"]
    visits = target["route_visits"]
    if target["required_waypoints"] != len(waypoints) or len(visits) != len(waypoints):
        raise AssertionError("The actual robot did not visit all required waypoints")
    if any(visits[index]["sequence"] >= visits[index + 1]["sequence"] for index in range(len(visits) - 1)):
        raise AssertionError("Physical waypoint visits are out of order")
    for visit, waypoint in zip(visits, waypoints, strict=True):
        if math.dist(visit["position_xy_m"], waypoint["target_xy_m"]) > waypoint["distance_m"]:
            raise AssertionError("Recorded waypoint pose exceeds its radius")
    if (not check["value"] or target["distance_xy_m"] > target["threshold_m"] or
            evidence["held_ticks"] < evidence["required_hold_ticks"]):
        raise AssertionError("Final physical goal or hold is incomplete")
    final_desks = [item for item in visible[-1]["targets"] if item["label"] == "desk" and
                   item["distance_status"] == "valid"]
    scene = next(event["detail"]["result"] for event in tools if event["detail"]["tool"] == "microduck.scene_info")
    position = evidence["robot_world_position_m"]
    desk_ids = {item["target_id"] for item in final_desks}
    distances = []
    for landmark in scene["public_map"]["nearby_landmarks"]:
        asset = PurePosixPath(landmark["prim_path"]).parent.name
        if f"/World/Environment/{asset}" not in desk_ids:
            continue
        minimum, maximum = landmark["world_min_m"], landmark["world_max_m"]
        nearest = [max(minimum[index], min(maximum[index], position[index])) for index in (0, 1)]
        distances.append(math.dist(position[:2], nearest))
    if not distances:
        raise AssertionError("Final visible desks lack authored geometry bounds")
    final_desk_distance = min(distances)
    if not 0.7 <= final_desk_distance <= 1.4:
        raise AssertionError("Final body-to-authored-desk-bound distance differs from the one-meter approach task")
    motion = {event["detail"]["result"]["metric_motion"]["request_id"]: event["detail"]["result"]["metric_motion"]
              for event in tools if event["detail"]["tool"] == "microduck.task_progress" and
              event["detail"]["result"].get("metric_motion")}
    for item in motion.values():
        if item["operation"] == "walk":
            start, position, yaw = item["start_position_m"], item["body_position_m"], item["start_yaw_rad"]
            projection = (position[0] - start[0]) * math.cos(yaw) + (position[1] - start[1]) * math.sin(yaw)
            destination = [start[0] + item["requested"] * math.cos(yaw), start[1] + item["requested"] * math.sin(yaw)]
            if (abs(projection - item["measured"]) > 1e-6 or
                    abs(math.dist(position[:2], destination) - item["error"]) > 1e-6):
                raise AssertionError("Walking measurements differ from actual recorded odometry")
        if item["completed"] and (item["error"] > item["tolerance"] or item["stopped_samples"] < 5):
            raise AssertionError("Completed motion lacks measured stopping and tolerance")
    distance = sum(abs(item["measured"]) for item in motion.values() if item["operation"] == "walk")
    if distance < 3:
        raise AssertionError("VLN lacks three meters of measured walking")
    print(json.dumps({"passed": True, "run_id": run["id"], "desk_observations": len(visible),
        "walk_calls": model_calls.count("microduck__walk"), "rotate_calls": model_calls.count("microduck__rotate"),
        "measured_walk_m": distance, "waypoint_visits": visits,
        "metric_phases": {key: item["phase"] for key, item in motion.items()},
        "completed_walks": sum(item["operation"] == "walk" and item["completed"] for item in motion.values()),
        "failed_walks": sum(item["operation"] == "walk" and item["phase"] == "failed" for item in motion.values()),
        "blocked_turns": sum(item["operation"] == "rotate" and item["phase"] == "blocked" for item in motion.values()),
        "final_desk_bound_distance_xy_m": final_desk_distance,
        "desk_bound_source": "authored static USD geometry bounds",
        "final_error_m": target["distance_xy_m"], "distance_source": "simulator_ground_truth",
        "checker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
