from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require_one(events: list[dict], event_type: str) -> dict:
    matches = [event for event in events if event["type"] == event_type]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {event_type}; found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an exported real EDH MicroDuck run")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--expected-verdict", choices=("passed", "failed"))
    parser.add_argument("--expected-stop-reason", choices=("policy_stop", "budget_exhausted"))
    parser.add_argument("--prior-export", type=Path)
    arguments = parser.parse_args()
    root = arguments.export_dir.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = json.loads((root / "source/events.json").read_text(encoding="utf-8"))
    frames = json.loads((root / "frames.json").read_text(encoding="utf-8"))
    if manifest["eventCount"] != len(events):
        raise AssertionError("Event count differs from the original export")
    if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
        raise AssertionError("Original event sequence has a gap")

    calls = [event["detail"]["data"]["name"] for event in events
             if event["type"] == "dsh.tool-call"]
    required_calls = {
        "microduck__read_sensor", "microduck__set_command",
        "microduck__task_progress",
        "execution__start", "execution__pause",
        "verification__check", "verification__submit",
    }
    if arguments.prior_export is None:
        required_calls.update(("microduck__policy_catalog", "microduck__select_policy",
                               "execution__resume"))
    if missing := required_calls - set(calls):
        raise AssertionError(f"Original trace lacks required tool calls: {sorted(missing)}")
    tool_results = [event["detail"] for event in events if event["type"] == "tool.completed"]
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
                          if item.get("tool") == "microduck.task_progress"]
        current_progress = [item["result"] for item in tool_results
                            if item.get("tool") == "microduck.task_progress"]
        if not prior_progress or not current_progress or (
            prior_progress[-1]["episode_id"], prior_progress[-1]["sequence"]
        ) != (
            current_progress[0]["episode_id"], current_progress[0]["sequence"]
        ) or any(item["episode_id"] != current_progress[0]["episode_id"] or
                 item["policy_name"] != "alpha_walking" for item in current_progress):
            raise AssertionError("Retained task did not continue its confirmed physical episode")
    if not any(item.get("tool") == "microduck.select_policy" and
               item["result"]["policy_name"] == "alpha_walking" for item in selection_results):
        raise AssertionError("Alpha Walking policy selection was not confirmed")
    if not any(item.get("tool") == "microduck.set_command" and
               item["result"]["command"]["twist"][0] > 0 for item in tool_results):
        raise AssertionError("A forward policy command was not confirmed")
    positions = [item["result"]["body_position_m"] for item in tool_results
                 if item.get("tool") == "microduck.task_progress"]
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

    observer = [frame for frame in frames if frame["kind"] == "simulation.frame"]
    if not observer:
        raise AssertionError("No physical observer frames were exported")
    previous_time = -1.0
    for frame in observer:
        if frame["image"]["name"] != "observer_follow.png" or (
            frame["image"]["width"], frame["image"]["height"]
        ) != (640, 480) or frame["nativeStepIndex"] != 4:
            raise AssertionError("Observer frame metadata differs from the native capture")
        if frame["simulationTimeS"] <= previous_time:
            raise AssertionError("Observer simulation time is not increasing")
        previous_time = frame["simulationTimeS"]
        image = (root / frame["file"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != frame["image"]["attachmentId"].removeprefix("sha256:"):
            raise AssertionError("Observer PNG differs from the native attachment")

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
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
