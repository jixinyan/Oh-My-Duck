import json

from jsonschema import Draft202012Validator


WAIT_ARGUMENTS = Draft202012Validator({"type": "object", "properties": {
    "max_wait_s": {"type": "integer", "minimum": 1, "maximum": 90}}, "additionalProperties": False})


def audit_motion_wait(events: list[dict]) -> dict:
    calls = [event for event in events if event["type"] == "dsh.tool-call" and
             event["detail"]["data"]["name"] == "microduck__wait_for_motion"]
    boundaries = [event for event in events if event["type"] == "execution.updated" and
                  event["detail"]["execution"]["state"] in {"paused", "ended"} and
                  event["detail"]["execution"]["device_confirmed"]]
    waits, pending, confirmed = [], [], []
    for event in events:
        if event["type"] != "tool.completed" or event["detail"]["tool"] != "microduck.wait_for_motion":
            continue
        detail, result = event["detail"], event["detail"]["result"]
        matching = [call for call in calls if call["sequence"] < event["sequence"] and
                    call["detail"]["assignmentId"] == detail["assignmentId"] and
                    call["detail"]["data"]["callId"] == detail["callId"]]
        if len(matching) != 1:
            raise AssertionError("Motion wait requires its original model tool call")
        arguments = json.loads(matching[0]["detail"]["data"]["arguments"])
        if not WAIT_ARGUMENTS.is_valid(arguments):
            raise AssertionError("Motion wait arguments differ from their native Schema")
        maximum = arguments.get("max_wait_s", 90)
        wait, execution = result["motion_wait"], result["execution"]
        boundary_confirmed = execution["state"] in {"paused", "ended"} and execution["device_confirmed"]
        if (type(wait["max_wait_s"]) is not int or wait["max_wait_s"] != maximum or
                type(wait["boundary_confirmed"]) is not bool or wait["boundary_confirmed"] != boundary_confirmed or
                type(result["sequence"]) is not int or result["sequence"] < 0 or
                type(result["stopped_samples"]) is not int or result["stopped_samples"] < 0 or
                result["goal_check"]["sequence"] != result["sequence"] or
                result["goal_check"]["episode_id"] != result["episode_id"]):
            raise AssertionError("Motion wait differs from its actual arguments or physical progress")
        motion = result["metric_motion"]
        if motion is not None and (motion["sequence"] != result["sequence"] or
                motion["body_position_m"] != result["body_position_m"] or
                motion["stopped_samples"] != result["stopped_samples"]):
            raise AssertionError("Motion wait requires one physical sequence for measured motion progress")
        if boundary_confirmed:
            if not any(boundary["sequence"] < event["sequence"] and
                all(boundary["detail"]["execution"][key] == execution[key]
                    for key in ("execution_id", "state", "device_confirmed")) and
                boundary["detail"]["execution"]["boundary_event_id"] == execution["boundary_id"]
                for boundary in boundaries):
                raise AssertionError("Motion wait lacks its original confirmed native boundary")
            confirmed.append(result["sequence"])
        else:
            if execution["state"] not in {"running", "pausing"}:
                raise AssertionError("Pending motion wait requires an active native execution")
            pending.append(result["sequence"])
        waits.append(maximum)
    if not pending or not confirmed:
        raise AssertionError("Motion wait acceptance requires active progress and a confirmed native boundary")
    return {"wait_results_verified": len(waits), "wait_windows_s": sorted(set(waits)),
            "pending_sequences": pending, "confirmed_sequences": confirmed}
