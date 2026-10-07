import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image, ImageStat

from accept_harness_replay import audit_motion_guard_pauses


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_navigation(root, configuration, minimum_distance):
    manifest = json.loads((root / "manifest.json").read_text())
    run = json.loads((root / "source/run.json").read_text())
    events = json.loads((root / "source/events.json").read_text())
    frames = json.loads((root / "frames.json").read_text())
    require(manifest["runId"] == run["id"] and manifest["runState"] == run["state"] == "succeeded",
            "Navigation requires matching native task success")
    require(manifest["eventCount"] == len(events) and
            [event["sequence"] for event in events] == list(range(1, len(events) + 1)),
            "Native event sequence is incomplete")
    calls = [event["detail"]["data"]["name"] for event in events if event["type"] == "dsh.tool-call"]
    require(calls.count("microduck__walk") >= 2 and calls.count("microduck__rotate") >= 2
            and "tasks__finish" in calls, "Navigation lacks actual model route tool calls")
    tools = [event["detail"] for event in events if event["type"] == "tool.completed"]
    scene = next(item["result"] for item in tools if item["tool"] == "microduck.scene_info")
    require(scene["scene_id"] == configuration["scene_id"] and scene["goal"] == configuration["goal"]
            and scene["solver"] == "Newton SolverMuJoCo", "Native scene differs from the declared benchmark")
    robot = configuration.get("robot_model", "allcollisions")
    policy = "roller" if robot == "groundcontact_rollers" else "alpha_walking"
    require(scene["robot_model"] == robot and scene["control_hz"] == 50
            and scene["physics_dt_s"] == 0.005, "Native robot or control timing differs")
    joints = [item["result"]["measurements"] for item in tools
              if item["tool"] == "microduck.read_sensor" and item["result"]["sensor"] == "joint_state"]
    require(joints and all(len(set(row["joint_names"])) == len(row["joint_position_rad"])
                          == len(row["joint_velocity_rad_s"]) == 14 for row in joints),
            "Route lacks actual fourteen-servo measurements")
    admissions = {item["result"]["request_id"]: item["result"] for item in tools
                  if item["tool"] in {"microduck.walk", "microduck.rotate"}}
    require(len(admissions) == sum(item["tool"] in {"microduck.walk", "microduck.rotate"} for item in tools),
            "Metric tool request identities are duplicated")
    for admission in admissions.values():
        require(admission["prepared"] and admission["policy_name"] == policy
                and admission["distance_tolerance_m"] == 0.05 and admission["angle_tolerance_deg"] == 5,
                "Route policy or official action timing differs")
    progress = [item["result"] for item in tools if item["tool"] == "microduck.task_progress"]
    require(len(progress) >= 3 and len({row["episode_id"] for row in progress}) == 1,
            "Navigation requires one continuous physical episode")
    require(all(row["contact_evidence"]["non_ground_external_contact_samples_total"] == 0
                for row in progress), "Recorded route has external obstacle contacts")
    pause_events = [event for event in events if not (
        event["type"] == "tool.completed" and event["detail"].get("tool") == "microduck.task_progress"
        and event["detail"]["result"]["execution"] is not None
        and event["detail"]["result"]["execution"]["state"] == "ended")]
    audit_motion_guard_pauses(pause_events, run["id"],
                             warmup_policy="roller" if robot == "groundcontact_rollers" else "velstand")
    endpoints = {}
    for row in progress:
        motion = row["metric_motion"]
        if motion is None or motion["phase"] in {"moving", "braking"}:
            continue
        admission = admissions[motion["request_id"]]
        require(motion["operation"] == admission["operation"] and motion["phase"] in {"complete", "failed", "blocked"}
                and motion["start_position_m"] == admission["start_position_m"]
                and motion["start_yaw_rad"] == admission["start_yaw_rad"],
                "Metric result differs from its original admitted target")
        require(row["sequence"] == motion["sequence"] and
                row["body_position_m"] == motion["body_position_m"], "Metric endpoint differs from current physics")
        start, position = motion["start_position_m"], row["body_position_m"]
        translation = math.dist(start[:2], position[:2])
        require(math.isclose(translation, motion["translation_xy_m"], abs_tol=1e-6),
                "Metric translation differs from recorded physical positions")
        if motion["operation"] == "walk":
            amount = admission["arguments"]["distance_m"]
            yaw = motion["start_yaw_rad"]
            target = [start[0] + amount * math.cos(yaw), start[1] + amount * math.sin(yaw)]
            error = math.dist(target, position[:2])
            require(math.isclose(error, motion["error"], abs_tol=1e-6),
                    "Walking error differs from the original physical target")
            tolerance = 0.05
        else:
            amount = admission["arguments"]["angle_deg"]
            measured = math.degrees(motion["unwrapped_yaw_rad"] - motion["start_yaw_rad"])
            error = abs(amount - measured)
            require(math.isclose(measured, motion["measured"], abs_tol=1e-6)
                    and math.isclose(error, motion["error"], abs_tol=1e-6),
                    "Rotation differs from the recorded accumulated physical yaw")
            tolerance = 5.0
        require(amount == motion["requested"] and motion["tolerance"] == tolerance,
                "Motion changed the requested amount or acceptance tolerance")
        if motion["completed"]:
            require(motion["phase"] == "complete" and error <= tolerance and not row["fallen"]
                    and row["stopped_samples"] >= 5 and motion["stopped_samples"] >= 5,
                    "Completed metric motion lacks measured upright stopping")
        endpoints[motion["request_id"]] = motion
    require(set(endpoints) == set(admissions), "An admitted metric action has no recorded terminal physical result")
    distance = sum(row["translation_xy_m"] for row in endpoints.values() if row["operation"] == "walk")
    require(distance >= minimum_distance, "Measured walking segment displacement is below the declared minimum")
    terminal = {event["detail"]["execution"]["execution_id"]: event["detail"]["execution"]
                for event in events if event["type"] == "execution.updated"
                and event["detail"]["execution"]["state"] == "ended"}
    require(terminal and all(row["device_confirmed"] and row["control_steps"] == row["policy_calls"]
                             and row["raw_sim_steps"] == 4 * row["control_steps"] for row in terminal.values()),
            "Native route execution counters or terminal confirmation differ")
    success = [event for event in events if event["type"] == "run.succeeded"]
    require(len(success) == 1, "Navigation has no unique native success event")
    verdict = next(event["detail"]["result"] for event in events
                   if event["type"] == "verification.completed"
                   and event["detail"]["result"]["verdict_id"] == success[0]["detail"]["verdictId"])
    final = progress[-1]
    execution = terminal[verdict["execution_id"]]
    require(verdict["status"] == "passed" and execution["stop_reason"] == "policy_stop"
            and execution["boundary_event_id"] == verdict["boundary_event_id"]
            and final["execution"]["execution_id"] == execution["execution_id"]
            and final["sequence"] == execution["control_steps"]
            and final["execution"]["device_confirmed"] and final["stopped_samples"] >= 5
            and not final["fallen"], "Final formal verdict lacks matching physical stopping")
    segment = final["command_segment"]
    command = next(item["result"] for item in tools if item["tool"] == "microduck.set_command"
                   and item["result"]["request_id"] == segment["request_id"])
    require(segment["used_control_steps"] >= 75 and not any(command["command"]["twist"]
            + command["command"]["head"] + command["command"]["body"])
            and final["command_block"] == [0] * 13,
            "Final navigation lacks 75 actual zero-command controls")
    finish = [item["result"] for item in tools if item["tool"] == "microduck.finish_policy"]
    require(finish and finish[-1]["accepted"] and finish[-1]["stop_confirmation"]["stopped_samples"] >= 5
            and finish[-1]["execution"] == execution,
            "Final navigation has no admitted measured policy stopping")
    check = next(item for item in verdict["checks"] if item["check_id"] == "goal_reached")
    physical = json.loads(check["reason"])
    evidence = physical["evidence"]
    require(physical["sequence"] == final["sequence"] and physical["episode_id"] == final["episode_id"]
            and evidence["robot_world_position_m"] == final["body_position_m"],
            "Formal terminal evidence differs from the final measured physical state")
    target, goal = evidence["target"], configuration["goal"]
    require(check["value"] is True and target["target_xy_m"] == goal["target_xy_m"]
            and target["threshold_m"] == goal["distance_m"]
            and math.isclose(math.dist(final["body_position_m"][:2], goal["target_xy_m"]),
                             target["distance_xy_m"], abs_tol=1e-6)
            and target["distance_xy_m"] <= goal["distance_m"]
            and evidence["required_hold_ticks"] == goal["hold_ticks"]
            and evidence["held_ticks"] >= goal["hold_ticks"], "Final native goal differs from the configured target")
    waypoints, visits = goal["waypoints"], target["route_visits"]
    require(len(waypoints) >= 2 and target["required_waypoints"] == len(visits) == len(waypoints),
            "Route lacks all declared ordered checkpoints")
    previous = -1
    for waypoint, visit in zip(waypoints, visits, strict=True):
        require(previous < visit["sequence"] <= final["sequence"] and
                math.dist(visit["position_xy_m"], waypoint["target_xy_m"]) <= waypoint["distance_m"],
                "Native checkpoint visit is out of order or outside its radius")
        previous = visit["sequence"]
    inspections = [item["result"] for item in tools if item["tool"] == "microduck.inspect_scene"]
    require(len({row["sequence"] for row in inspections}) >= 3 and all(
        row["distance_source"] == "simulator_ground_truth" for row in inspections),
        "Navigation lacks refreshed, explicitly sourced observations")
    attachments = {row["image"]["attachmentId"] for row in frames}
    require(all(row["image_ref"]["attachmentId"] in attachments for row in inspections),
            "Navigation perception has no original exported image")
    observers = 0
    last_time = -1
    for frame in frames:
        path = (root / frame["file"]).resolve(strict=True)
        require(path.is_relative_to(root), "Native frame leaves the export directory")
        pixels = path.read_bytes()
        require(hashlib.sha256(pixels).hexdigest() == frame["image"]["attachmentId"].removeprefix("sha256:")
                and len(pixels) == frame["image"]["bytes"], "Frame differs from the native image attachment")
        with Image.open(BytesIO(pixels)) as image:
            require(image.size == (frame["image"]["width"], frame["image"]["height"])
                    and max(ImageStat.Stat(image).stddev) > 0, "Native image dimensions or pixels are invalid")
        if frame["kind"] == "simulation.frame" and frame["image"]["name"] == "observer_follow.png":
            require(frame["nativeStepIndex"] == 4 and frame["simulationTimeS"] > last_time,
                    "Observer frames differ from the actual physical cadence")
            last_time = frame["simulationTimeS"]
            observers += 1
    require(observers >= 10, "Route has insufficient native observer frames")
    return {"status": "passed", "run_id": run["id"], "scene_id": configuration["scene_id"],
            "policy": policy, "measured_walk_segment_displacement_m": distance,
            "minimum_walk_segment_displacement_m": minimum_distance, "waypoint_visits": visits,
            "final_error_m": target["distance_xy_m"], "stopped_samples": final["stopped_samples"],
            "observer_frames": observers, "observations": len(inspections), "verdict_id": verdict["verdict_id"],
            "metric_results": list(endpoints.values()), "executions": list(terminal.values()),
            "non_ground_external_contact_samples": final["contact_evidence"]["non_ground_external_contact_samples_total"],
            "walk_calls": calls.count("microduck__walk"), "rotate_calls": calls.count("microduck__rotate"),
            "models": run["configuration"]["models"],
            "solver_settings": {key: scene[key] for key in
                                ("solver", "solver_iterations", "solver_ls_iterations", "physics_cuda_graph")},
            "source_artifacts_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                        for name in ("manifest.json", "source/run.json", "source/events.json", "frames.json")},
            "scope": "GT-assisted fixed-scene native navigation; walk segment displacement is not continuous path length"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--minimum-distance-m", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(math.isfinite(args.minimum_distance_m) and args.minimum_distance_m > 0,
            "Minimum measured distance must be positive and finite")
    report = verify_navigation(args.export.resolve(strict=True),
                               json.loads(args.scene_config.read_text()), args.minimum_distance_m)
    report["scene_config_sha256"] = hashlib.sha256(args.scene_config.read_bytes()).hexdigest()
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "run_id", "scene_id", "final_error_m")}))


if __name__ == "__main__":
    main()
