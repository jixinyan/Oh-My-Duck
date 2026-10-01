import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--robot-model", required=True)
    parser.add_argument("--stop-reason", choices=("policy_stop", "episode_terminated"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.export.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    run = json.loads((root / "source/run.json").read_text())
    events = json.loads((root / "source/events.json").read_text())
    frames = json.loads((root / "frames.json").read_text())
    if manifest["runId"] != run["id"] or manifest["runState"] != run["state"] or run["state"] != "succeeded":
        raise AssertionError("Showcase lacks native task success")
    if len(events) != manifest["eventCount"] or [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
        raise AssertionError("Native event sequence differs from the exported manifest")
    tools = [event["detail"] for event in events if event["type"] == "tool.completed"]
    selections = [item["result"] for item in tools if item["tool"] in
                  {"microduck.select_policy", "microduck.transition_policy", "microduck.walk", "microduck.rotate"}]
    for selection in selections:
        if "action_spec" in selection:
            spec = selection["action_spec"]
            if (spec["robot_model"] != args.robot_model or len(spec["joint_names"]) != 14
                    or spec["control_hz"] != 50 or spec["physics_steps_per_action"] != 4 or spec["physics_dt_s"] != 0.005):
                raise AssertionError("Policy selection differs from the actual robot and canonical control timing")
    progress = [item["result"] for item in tools if item["tool"] == "microduck.task_progress"]
    for policy in args.policies:
        if not any(row["policy_name"] == policy and row["sequence"] > 0 for row in progress):
            raise AssertionError(f"Actual physical progress missing for policy {policy}")
    if len(progress) < 2 or math.dist(progress[0]["body_position_m"][:2], progress[-1]["body_position_m"][:2]) < 0.1:
        raise AssertionError("Showcase lacks measured locomotion")
    if progress[-1]["fallen"] or progress[-1]["stopped_samples"] < 5:
        raise AssertionError("Final progress lacks measured upright stopping")
    terminal = {}
    for event in events:
        if event["type"] == "execution.updated":
            execution = event["detail"]["execution"]
            if execution["state"] == "ended":
                terminal[execution["execution_id"]] = execution
    if not terminal:
        raise AssertionError("No native terminal execution boundary")
    for execution in terminal.values():
        if (not execution["device_confirmed"] or execution["raw_sim_steps"] != execution["control_steps"] * 4
                or execution["policy_calls"] != execution["control_steps"]):
            raise AssertionError("Native execution timing or boundary confirmation differs")
    successful = [event["detail"]["result"] for event in events if event["type"] == "verification.completed"
                  and event["detail"]["result"]["status"] == "passed"]
    successes = [event for event in events if event["type"] == "run.succeeded"]
    if len(successes) != 1 or not successful:
        raise AssertionError("Native independent Verifier has no successful final result")
    verdict = next(item for item in successful if item["verdict_id"] == successes[0]["detail"]["verdictId"])
    execution = terminal[verdict["execution_id"]]
    if execution["boundary_event_id"] != verdict["boundary_event_id"] or execution["stop_reason"] != args.stop_reason:
        raise AssertionError("Final verdict differs from the admitted stop boundary")
    final_progress = progress[-1]
    if (final_progress["execution"]["execution_id"] != execution["execution_id"]
            or final_progress["execution"]["boundary_id"] != verdict["boundary_event_id"]
            or not final_progress["execution"]["device_confirmed"]):
        raise AssertionError("Final physical progress differs from the verified execution boundary")
    goal_check = next(item for item in verdict["checks"] if item["check_id"] == "goal_reached")
    if goal_check["value"] is not True:
        raise AssertionError("Formal native goal check did not pass")
    evidence = json.loads(goal_check["reason"])["evidence"]
    if (evidence["target"]["distance_xy_m"] > evidence["target"]["threshold_m"]
            or evidence["held_ticks"] < evidence["required_hold_ticks"]):
        raise AssertionError("Final native geometry and hold differ from the formal verdict")
    if args.stop_reason == "policy_stop":
        finish = [item for item in tools if item["tool"] == "microduck.finish_policy"]
        if not finish or finish[-1]["result"]["stop_confirmation"]["stopped_samples"] < 5:
            raise AssertionError("Physical stop was not admitted before verification")
    else:
        selection_event = next(event for event in reversed(events) if event["type"] == "tool.completed"
            and event["detail"]["tool"] == "microduck.select_policy"
            and event["detail"]["result"]["policy_name"] == final_progress["policy_name"])
        selected = selection_event["detail"]["result"]
        preceding = next(event["detail"]["result"] for event in reversed(events)
            if event["sequence"] < selection_event["sequence"] and event["type"] == "tool.completed"
            and event["detail"]["tool"] == "microduck.task_progress")
        if (selected["kind"] != "episodic" or selected["encoding"] != "phase"
                or not math.isclose(final_progress["sequence"] - preceding["sequence"], selected["duration_s"] * 50)
                or final_progress["command_segment"]["used_control_steps"] < 75
                or any(final_progress["motion_guard"]["command"]["twist"])):
            raise AssertionError("Episodic termination lacks measured full duration and final zero-command control")
    observer_count = 0
    previous_time = -1
    for frame in frames:
        path = (root / frame["file"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise AssertionError("Frame path leaves the export directory")
        pixels = path.read_bytes()
        if hashlib.sha256(pixels).hexdigest() != frame["image"]["attachmentId"].removeprefix("sha256:"):
            raise AssertionError("Frame differs from the original native attachment")
        with Image.open(path) as image:
            if image.size != (frame["image"]["width"], frame["image"]["height"]) or np.asarray(image).std() <= 0:
                raise AssertionError("Native image has invalid dimensions or uniform pixels")
        if frame["kind"] == "simulation.frame" and frame["image"]["name"] == "observer_follow.png":
            if frame["nativeStepIndex"] != 4 or frame["simulationTimeS"] <= previous_time:
                raise AssertionError("Observer timing differs from actual Newton controls")
            previous_time = frame["simulationTimeS"]
            observer_count += 1
    if observer_count < 10:
        raise AssertionError("Actual observer recording is incomplete")
    model_results = [item["result"] for item in tools if item["tool"] == "microduck.inspect_scene"
                     and item["result"]["detection_source"] == "sam3.1"]
    if not model_results or not all("sam3.1" in result["models"] and "yolo26" in result["models"] for result in model_results):
        raise AssertionError("Actual joint SAM3.1 and YOLO26 inference missing")
    wheel_samples = [item["result"] for item in tools if item["tool"] == "microduck.read_sensor"
        and item["result"].get("sensor") == "joint_state"]
    if args.robot_model == "robot_groundcontact_rollers":
        wheel_names = {"passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"}
        if not wheel_samples or any(set(row["measurements"]["passive_joints"]) != wheel_names for row in wheel_samples):
            raise AssertionError("Actual passive wheel measurements are missing")
        if not any(all(abs(wheel["velocity_rad_s"]) > 0.1
                       for wheel in row["measurements"]["passive_joints"].values()) for row in wheel_samples):
            raise AssertionError("No measured rotation of all four passive wheels")
    metric_failures = {row["metric_motion"]["request_id"]: row["metric_motion"] for row in progress
        if row["metric_motion"] is not None and row["metric_motion"]["phase"] == "failed"}
    report = {"status": "passed", "run_id": run["id"], "policies_with_physical_progress": args.policies,
              "native_executions": list(terminal.values()), "observer_frames": observer_count,
              "model_observations": len(model_results), "valid_model_targets": sum(target["distance_status"] == "valid"
                  for result in model_results for target in result["targets"]),
              "final_native_evidence": evidence, "verdict_id": verdict["verdict_id"],
              "stop_reason": args.stop_reason, "final_stopped_samples": final_progress["stopped_samples"],
              "failed_metric_commands": list(metric_failures.values()),
              "non_ground_external_contact_samples": final_progress["contact_evidence"]["non_ground_external_contact_samples_total"],
              "passive_wheel_observations": len(wheel_samples) if args.robot_model == "robot_groundcontact_rollers" else 0,
              "scope": "actual policy progress, native boundaries, image provenance and formal navigation; object effects require separate measurements",
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
