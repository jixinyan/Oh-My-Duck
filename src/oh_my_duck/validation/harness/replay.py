from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

PROGRESS_TOOLS = {"microduck.task_progress", "microduck.observe", "microduck.wait_for_motion"}

def require_one(events: list[dict], event_type: str) -> dict:
    matches = [event for event in events if event["type"] == event_type]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {event_type}; found {len(matches)}")
    return matches[0]


def audit_motion_guard_pauses(events: list[dict], run_id: str, warmup_policy: str = "velstand") -> None:
    completed_commands = [event for event in events if event["type"] == "tool.completed" and
                          event["detail"].get("tool") == "microduck.set_command"]
    commands = {event["detail"]["result"]["request_id"]: event
                for event in completed_commands}
    metric_commands = {event["detail"]["result"]["request_id"]: event for event in events
                       if event["type"] == "tool.completed" and
                       event["detail"].get("tool") in ("microduck.walk", "microduck.rotate")}
    if len(commands) != len(completed_commands):
        raise AssertionError("Physical command request identities are duplicated")
    guarded = [event for event in events if event["type"] == "tool.completed" and
               event["detail"].get("tool") in PROGRESS_TOOLS and
               event["detail"]["result"].get("motion_guard")]
    if not guarded:
        raise AssertionError("Native automatic pause lacks measured motion-guard progress")
    boundaries = [event for event in events if event["type"] == "execution.updated" and
                  event["detail"]["execution"]["state"] in ("paused", "ended") and
                  event["detail"]["execution"]["device_confirmed"]]
    for progress_event in guarded:
        progress = progress_event["detail"]["result"]
        guard = progress["motion_guard"]
        execution = progress["execution"]
        if (guard["run_task_id"] != run_id or
                guard["execution_id"] != execution["execution_id"] or
                guard["episode_id"] != progress["episode_id"] or
                guard["sequence"] != progress["sequence"] or
                execution["state"] not in ("paused", "ended") or not execution["device_confirmed"]):
            raise AssertionError("Motion guard differs from its confirmed physical boundary")
        matching = [event for event in boundaries if
                   datetime.fromisoformat(event["detail"]["execution"]["boundary_at"].replace("Z", "+00:00")) <=
                   datetime.fromisoformat(progress_event["at"].replace("Z", "+00:00")) and
                   event["detail"]["execution"]["execution_id"] == execution["execution_id"] and
                   event["detail"]["execution"]["state"] == execution["state"] and
                   event["detail"]["execution"]["boundary_event_id"] == execution["boundary_id"] and
                   event["detail"]["execution"]["task_scope"]["task_id"] == run_id]
        if not matching:
            raise AssertionError("Motion guard lacks its matching native confirmed boundary")
        generation_delta = 1
        if execution["state"] == "ended":
            generation_delta = 2
            if execution["stop_reason"] != "policy_stop":
                raise AssertionError("Retained motion guard requires a confirmed policy stop")
            terminal = matching[-1]["detail"]["execution"]
            if not any(event["sequence"] < matching[-1]["sequence"] and
                       event["detail"]["execution"]["state"] == "paused" and
                       event["detail"]["execution"]["execution_id"] == execution["execution_id"] and
                       event["detail"]["execution"]["task_scope"]["task_id"] == run_id and
                       all(event["detail"]["execution"][key] == terminal[key]
                           for key in ("control_steps", "raw_sim_steps", "policy_calls"))
                       for event in boundaries):
                raise AssertionError("Terminal guard lacks its preceding confirmed physical pause")
        if execution["generation"] != guard["generation"] + generation_delta:
            raise AssertionError("Motion guard differs from the native stop generation")
        command_event = commands.get(guard["command_request_id"])
        if command_event is None and guard.get("metric_request_id") is None:
            segment = progress.get("command_segment", {})
            initial_start = [event for event in events if event["type"] == "tool.completed" and
                             event["detail"].get("tool") == "execution.start" and
                             event["detail"]["result"]["execution"]["execution_id"] == execution["execution_id"] and
                             event["sequence"] < progress_event["sequence"]]
            if (len(initial_start) == 1 and segment.get("request_id") == guard["command_request_id"] and
                    segment.get("effective_after_sequence") == 0 and
                    segment.get("max_control_steps") == segment.get("used_control_steps") == 75 and
                    guard["reason"] == "command_segment_complete" and
                    guard["sequence"] == guard["used_control_steps"] == guard["max_control_steps"] == 75 and
                    progress["policy_name"] == warmup_policy and progress["stopped_samples"] >= 5 and
                    progress["command_block"] == [0] * 13 and
                    guard["command"]["twist"] == [0, 0, 0] and
                    guard["command"]["head"] == [0] * 4 and guard["command"]["body"] == [0] * 6 and
                    not any(event["sequence"] < progress_event["sequence"] for event in
                            [*commands.values(), *metric_commands.values()])):
                continue
        if command_event is None and guard.get("metric_request_id") in metric_commands:
            metric_event = metric_commands[guard["metric_request_id"]]
            if metric_event["sequence"] >= progress_event["sequence"]:
                raise AssertionError("Metric motion admission follows its physical progress")
            command = guard["command_admission"]
            if command["request_id"] != guard["command_request_id"]:
                raise AssertionError("Metric motion command identity differs from its pause")
            if (guard["sequence"] != command["effective_after_sequence"] + guard["used_control_steps"] or
                    guard["command"] != command["command"] or
                    not 0 < guard["used_control_steps"] <= command["max_control_steps"]):
                raise AssertionError("Metric motion pause differs from its admitted physical command")
            motion = progress["metric_motion"]
            if (motion["request_id"] != guard["metric_request_id"] or motion["sequence"] != progress["sequence"] or
                    motion["completed"] and (motion["error"] > motion["tolerance"] or motion["stopped_samples"] < 5)):
                raise AssertionError("Metric completion lacks measured goal and stop evidence")
            continue
        if command_event is None or command_event["sequence"] >= progress_event["sequence"]:
            raise AssertionError("Motion guard lacks its completed physical command")
        command = command_event["detail"]["result"]
        if (guard["sequence"] != command["effective_after_sequence"] +
                guard["used_control_steps"] or
                not 0 < guard["used_control_steps"] <= guard["max_control_steps"] or
                guard["max_control_steps"] != command["max_control_steps"] or
                guard["command"] != command["command"]):
            raise AssertionError("Motion guard action count differs from its bounded command")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an exported real EDH MicroDuck run")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--expected-verdict", choices=("passed", "failed"))
    parser.add_argument("--expected-stop-reason", choices=("policy_stop", "budget_exhausted"))
    parser.add_argument("--prior-export", type=Path)
    parser.add_argument("--require-stop-progress", action="store_true")
    parser.add_argument("--require-metric-tools", action="store_true")
    arguments = parser.parse_args()
    root = arguments.export_dir.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((root / "source/run.json").read_text(encoding="utf-8"))
    environment = run["configuration"]["launchProfile"]["environment"]
    if environment.startswith("Isaac Lab Newton/BAM:"):
        observer_size = (1280, 720)
    elif environment == "CPU MuJoCo/BAM official 8 × 6 m apartment":
        observer_size = (640, 480)
    else:
        raise AssertionError("Run environment has no accepted observer specification")
    events = json.loads((root / "source/events.json").read_text(encoding="utf-8"))
    frames = json.loads((root / "frames.json").read_text(encoding="utf-8"))
    if manifest["eventCount"] != len(events):
        raise AssertionError("Event count differs from the original export")
    if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
        raise AssertionError("Original event sequence has a gap")

    calls = [event["detail"]["data"]["name"] for event in events
             if event["type"] == "dsh.tool-call"]
    required_calls = {
        "microduck__set_command",
        "execution__start",
        "verification__check", "verification__submit",
    }
    if not set(calls) & {"microduck__read_sensor", "microduck__observe"} or not set(calls) & {
            name.replace(".", "__") for name in PROGRESS_TOOLS}:
        raise AssertionError("Original trace lacks actual sensor and measured-progress tool calls")
    if arguments.prior_export is None:
        required_calls.update(("microduck__policy_catalog", "execution__resume"))
        if not set(calls) & {"microduck__select_policy", "microduck__walk", "microduck__rotate"}:
            raise AssertionError("Original trace lacks an official policy selection tool")
    if arguments.require_metric_tools:
        required_calls.discard("microduck__select_policy")
        required_calls.add("microduck__walk")
    if missing := required_calls - set(calls):
        raise AssertionError(f"Original trace lacks required tool calls: {sorted(missing)}")
    if "execution__pause" not in calls:
        audit_motion_guard_pauses(events, manifest["runId"])
    tool_results = [event["detail"] for event in events if event["type"] == "tool.completed"]
    if arguments.require_metric_tools:
        prepared = {item["result"]["request_id"]: item["result"] for item in tool_results
                    if item["tool"] == "microduck.walk"}
        completed_motion = [item["result"] for item in tool_results if item["tool"] in PROGRESS_TOOLS
                            and item["result"].get("metric_motion", {}) is not None
                            and item["result"].get("metric_motion", {}).get("completed") is True]
        walking = [progress for progress in completed_motion if progress["metric_motion"]["operation"] == "walk"]
        if not walking:
            raise AssertionError("Metric walking lacks a completed measured target")
        for progress in walking:
            motion = progress["metric_motion"]
            admission = prepared[motion["request_id"]]
            requested = admission["arguments"]["distance_m"]
            start, yaw = motion["start_position_m"], motion["start_yaw_rad"]
            target = [start[0] + requested * math.cos(yaw), start[1] + requested * math.sin(yaw)]
            native_position = progress["body_position_m"]
            native_error = math.dist(target, native_position[:2])
            if (motion["requested"] != requested or motion["unit"] != "m" or
                    math.dist(native_position, motion["body_position_m"]) > 1e-6 or
                    abs(native_error - motion["error"]) > 1e-6 or native_error > 0.05 or
                    progress["stopped_samples"] < 5 or progress["fallen"]):
                raise AssertionError("Metric walking differs from its measured native target and stop")
    stop_progress_count = 0
    if arguments.require_stop_progress:
        samples_by_sequence = {}
        last_progress = None
        for detail in tool_results:
            if detail["tool"] in ("microduck.select_policy", "microduck.transition_policy"):
                samples_by_sequence.clear()
            if detail["tool"] in PROGRESS_TOOLS:
                result = detail["result"]
                samples = result["stopped_samples"]
                if (type(samples) is not int or samples < 0 or
                        result["required_stopped_samples"] != 5):
                    raise AssertionError("Stop progress lacks measured nonnegative sample counts")
                identity = (result["episode_id"], result["sequence"], result["policy_name"])
                if identity in samples_by_sequence and samples_by_sequence[identity] != samples:
                    raise AssertionError("Stop samples changed without a physical control step")
                samples_by_sequence[identity] = samples
                last_progress = result
                stop_progress_count += 1
            if detail["tool"] == "microduck.finish_policy":
                result = detail["result"]
                if (last_progress is None or last_progress["stopped_samples"] < 5 or
                        result["stop_confirmation"]["stopped_samples"] !=
                        last_progress["stopped_samples"]):
                    raise AssertionError("Policy finish differs from measured stop progress")
        if stop_progress_count < 2:
            raise AssertionError("Stop progress lacks observations across physical execution")
    if arguments.prior_export is None:
        selection_results = tool_results
    else:
        prior_root = arguments.prior_export.resolve(strict=True)
        prior_run = json.loads((prior_root / "source/run.json").read_text(encoding="utf-8"))
        current_run = json.loads((root / "source/run.json").read_text(encoding="utf-8"))
        if prior_run["userSessionId"] != current_run["userSessionId"] or (
            json.loads((prior_root / "manifest.json").read_text(encoding="utf-8"))["runId"] ==
            manifest["runId"]
        ):
            raise AssertionError("Retained physical task does not belong to the prior session")
        prior_events = json.loads((prior_root / "source/events.json").read_text(encoding="utf-8"))
        selection_results = [event["detail"] for event in prior_events
                             if event["type"] == "tool.completed"]
        prior_progress = [item["result"] for item in selection_results
                          if item.get("tool") in PROGRESS_TOOLS]
        current_progress = [item["result"] for item in tool_results
                            if item.get("tool") in PROGRESS_TOOLS]
        if not prior_progress or not current_progress or (
            prior_progress[-1]["episode_id"], prior_progress[-1]["sequence"]
        ) != (
            current_progress[0]["episode_id"], current_progress[0]["sequence"]
        ) or any(item["episode_id"] != current_progress[0]["episode_id"] or
                 item["policy_name"] != "alpha_walking" for item in current_progress):
            raise AssertionError("Retained task did not continue its confirmed physical episode")
    if not any(item.get("tool") == "microduck.select_policy" and
               item["result"]["policy_name"] == "alpha_walking" or
               item.get("tool") == "microduck.walk" and item["result"]["policy_name"] == "alpha_walking"
               for item in selection_results):
        raise AssertionError("Alpha Walking policy selection was not confirmed")
    if not any(item.get("tool") == "microduck.set_command" and
               item["result"]["command"]["twist"][0] > 0 or
               item.get("tool") == "microduck.walk" and item["result"]["arguments"]["distance_m"] > 0
               for item in tool_results):
        raise AssertionError("A forward policy command was not confirmed")
    positions = [item["result"]["body_position_m"] for item in tool_results
                 if item.get("tool") in PROGRESS_TOOLS]
    if len(positions) < 2 or not math.dist(positions[0][:2], positions[-1][:2]) > 0.01:
        raise AssertionError("Recorded physical position did not change")

    terminal = [event for event in events if event["type"] == "execution.updated" and
                event["detail"]["execution"]["state"] == "ended"]
    if len(terminal) != 1 or not terminal[0]["detail"]["execution"]["device_confirmed"]:
        raise AssertionError("Native ActionGate did not confirm its terminal boundary")
    execution = terminal[0]["detail"]["execution"]
    if execution["stop_reason"] not in ("policy_stop", "budget_exhausted"):
        raise AssertionError("Execution ended outside a verified policy boundary")
    if (arguments.expected_stop_reason is not None and
            execution["stop_reason"] != arguments.expected_stop_reason):
        raise AssertionError("Execution stop reason differs from the requested result")
    if execution["stop_reason"] == "policy_stop" and not any(
        item.get("tool") == "microduck.finish_policy" and item["result"]["accepted"] is True
        for item in tool_results
    ):
        raise AssertionError("Policy finish was not admitted")
    checked = require_one(events, "verification.checked")
    completed = require_one(events, "verification.completed")
    verdict = completed["detail"]["result"]
    if verdict["execution_id"] != execution["execution_id"] or (
        verdict["boundary_event_id"] != execution["boundary_event_id"]
    ):
        raise AssertionError("Verifier result differs from the stopped execution")
    facts = checked["detail"]["facts"]
    if len(facts) != 1 or len(verdict["checks"]) != 1 or (
        facts[0]["check_id"] != "goal_reached" or
        verdict["checks"][0]["check_id"] != "goal_reached" or
        facts[0]["value"] != verdict["checks"][0]["value"]
    ):
        raise AssertionError("Native goal check differs from the formal verdict")
    if verdict["status"] not in ("passed", "failed") or (
        verdict["checks"][0]["value"] is not (verdict["status"] == "passed")
    ):
        raise AssertionError("Formal verdict differs from the native goal result")
    if arguments.expected_verdict is not None and verdict["status"] != arguments.expected_verdict:
        raise AssertionError("Formal verdict differs from the requested result")

    observer = [frame for frame in frames if frame["kind"] == "simulation.frame"
                and frame["image"]["name"] == "observer_follow.png"]
    if not observer:
        raise AssertionError("No physical observer frames were exported")
    previous_time = -1.0
    for frame in observer:
        if frame["image"]["name"] != "observer_follow.png" or (
            frame["image"]["width"], frame["image"]["height"]
        ) != observer_size or frame["nativeStepIndex"] != 4:
            raise AssertionError("Observer frame metadata differs from the native capture")
        if frame["simulationTimeS"] <= previous_time:
            raise AssertionError("Observer simulation time is not increasing")
        previous_time = frame["simulationTimeS"]
        image = (root / frame["file"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != frame["image"]["attachmentId"].removeprefix("sha256:"):
            raise AssertionError("Observer PNG differs from the native attachment")
        with Image.open(root / frame["file"]) as captured:
            if captured.format != "PNG" or captured.size != observer_size:
                raise AssertionError("Observer PNG dimensions differ from native metadata")
            captured.verify()

    if verdict["status"] == "passed":
        if manifest["runState"] != "succeeded" or (
            require_one(events, "run.succeeded")["detail"]["verdictId"] != verdict["verdict_id"]
        ) or "tasks__finish" not in calls:
            raise AssertionError("Task success differs from the formal verdict")
    elif manifest["runState"] != "failed" or (
        require_one(events, "run.abandoned")["detail"]["status"] != "failed"
    ) or "tasks__abandon" not in calls:
        raise AssertionError("Task failure differs from the formal verdict")
    print(json.dumps({
        "run_id": manifest["runId"], "events": len(events),
        "observer_frames": len(observer), "control_steps": execution["control_steps"],
        "raw_sim_steps": execution["raw_sim_steps"], "stop_reason": execution["stop_reason"],
        "verdict_id": verdict["verdict_id"], "goal_reached": verdict["checks"][0]["value"],
        "formal_status": verdict["status"], "run_state": manifest["runState"],
        "environment": environment, "observer_size": observer_size,
        "audited_stop_progress_count": stop_progress_count,
        "strict_metric_targets_required": arguments.require_metric_tools,
        "checker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
