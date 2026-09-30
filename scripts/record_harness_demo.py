import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "unknown"}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def request(origin: str, route: str, body=None):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    query = Request(origin + route, data=data,
                    headers={"Content-Type": "application/json"} if data is not None else {})
    with urlopen(query, timeout=240) as response:
        return json.load(response)


def tool_status(event: dict) -> dict:
    result = event["detail"].get("result", {})
    summary = {key: result[key] for key in ("sequence", "body_position_m", "policy_name",
               "command", "max_control_steps", "body_twist", "fallen", "sensor",
               "stopped_samples", "required_stopped_samples") if key in result}
    if event["type"] == "tool.failed":
        summary["error"] = event["detail"]["error"]
    if guard := result.get("motion_guard"):
        summary["motion_guard"] = {key: guard[key] for key in
                                   ("reason", "sequence", "used_control_steps", "central_tof_closest_mm")}
    return {"sequence": event["sequence"], "tool": event["detail"]["tool"], "result": summary}


def history(origin: str, run_id: str, count: int, after: int = 0) -> list[dict]:
    events = []
    while after + len(events) < count:
        cursor = after + len(events)
        page = request(origin, f"/api/runs/{run_id}/history?after={cursor}&through={count}")
        if (page["runId"] != run_id or page["afterSequence"] != cursor or
                page["eventTotal"] != count or not page["events"]):
            raise ValueError("Native event pagination differs from the requested record")
        for event in page["events"]:
            if event["sequence"] != after + len(events) + 1:
                raise ValueError("Native event history contains a sequence gap")
            events.append(event)
        if page["throughSequence"] != after + len(events):
            raise ValueError("Native event page end differs from its contents")
    return events


def export_run(origin: str, run_id: str, output: Path) -> dict:
    run_route = f"/api/runs/{quote(run_id, safe='')}?events=none"
    run = request(origin, run_route)
    if run["id"] != run_id or run["state"] not in TERMINAL_STATES:
        raise ValueError("Replay export requires the requested terminal run")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    events = history(origin, run_id, run["eventCount"])
    if request(origin, run_route) != run:
        raise ValueError("Native run changed during replay export")
    save_json(source / "run.json", run)
    save_json(source / "events.json", events)
    rows = []
    known = set()
    for event in events:
        if event["type"] == "simulation.frame":
            sample = event["detail"]["sample"]
        elif event["type"] == "tool.completed" and event["detail"].get("tool") == "perception.capture":
            sample = event["detail"]["result"]
        else:
            continue
        for image in sample.get("images", []):
            evidence_id = sample["evidence"]["id"]
            attachment = image["attachmentId"]
            identity = (evidence_id, attachment)
            if identity in known:
                continue
            known.add(identity)
            frame = event["type"] == "simulation.frame"
            folder = "frames" if frame else "observations"
            route = (f"replay/frames/{event['sequence']}/images/{quote(attachment, safe='')}" if frame
                     else f"evidence/{quote(evidence_id, safe='')}/images/{quote(attachment, safe='')}")
            with urlopen(origin + f"/api/runs/{run_id}/" + route, timeout=120) as response:
                data = response.read()
            if (len(data) != image["bytes"] or not attachment.startswith("sha256:") or
                    hashlib.sha256(data).hexdigest() != attachment[7:]):
                raise ValueError("Native image identity or byte count differs")
            path = Path(folder) / f"{event['sequence']}-{uuid4().hex}-{Path(image['name']).name}"
            (output / folder).mkdir(exist_ok=True)
            (output / path).write_bytes(data)
            detail = event["detail"]
            rows.append({
                "kind": "simulation.frame" if frame else "agent.observation",
                "eventSequence": event["sequence"], "eventAt": event["at"],
                "evidenceId": evidence_id, "observedAt": sample["evidence"].get("observed_at"),
                "sampleSequence": sample.get("sequence"), "image": image,
                "executionId": detail.get("executionId"),
                "policyRequestId": detail.get("policyRequestId"), "segmentId": detail.get("segmentId"),
                "nativeStepIndex": detail.get("nativeStepIndex"),
                "simulationTimeS": detail.get("simulationTimeS"),
                "file": path.as_posix(), "availability": "available",
            })
    save_json(output / "frames.json", rows)
    manifest = {
        "schemaVersion": "edh.run_replay.v1", "runId": run_id, "runState": run["state"],
        "source": run["source"], "task": run["scenario"], "instruction": run["instruction"],
        "eventCount": len(events), "imageCount": len(rows), "verdicts": run["verdicts"],
        "runError": run.get("error"), "videos": [],
        "originalRecords": ["source/run.json", "source/events.json", "frames.json"],
        "exporterSourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Record actual native Harness sessions and replay media")
    parser.add_argument("operation", choices=("open", "task", "status", "export", "close"))
    parser.add_argument("--base-url", default="http://127.0.0.1:4338")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--scenario")
    parser.add_argument("--instruction", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("Harness endpoint must be an HTTP(S) origin")
    origin = args.base_url.rstrip("/")
    if args.operation == "open":
        if args.session is None or args.profile is None or args.session.exists():
            raise ValueError("Session open requires a profile and a new session file")
        result = request(origin, "/api/sessions", {"profileId": args.profile, "requestId": str(uuid4())})
        args.session.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.session, result)
    elif args.operation == "export":
        if args.output is None or args.run_id is None:
            raise ValueError("Export requires a run ID and a new output directory")
        result = export_run(origin, args.run_id, args.output)
    elif args.operation == "status":
        if args.run_id is None:
            raise ValueError("Status requires a run ID")
        run = request(origin, f"/api/runs/{quote(args.run_id, safe='')}?events=none")
        result = {name: run.get(name) for name in ("id", "state", "updatedAt", "eventCount", "error", "verdicts")}
        result["executions"] = [{key: execution.get(key) for key in
                                 ("execution_id", "state", "control_steps", "raw_sim_steps",
                                  "device_confirmed", "stop_reason")}
                                for execution in run["executions"]]
        count = run["eventCount"]
        events = history(origin, args.run_id, count, max(0, count - 200))
        result["recent_tools"] = [tool_status(event)
                                  for event in events if event["type"] in
                                  ("tool.completed", "tool.failed")][-8:]
        outputs = [event for event in events if event["type"] == "agent.output"]
        for event in reversed(outputs):
            texts = [block["text"] for block in event["detail"].get("message", {}).get("content", [])
                     if block["type"] == "text"]
            if texts:
                result["latest_agent_text"] = texts
                break
    else:
        if args.session is None:
            raise ValueError("Task and close require the recorded session file")
        session = json.loads(args.session.read_text(encoding="utf-8"))
        route = f"/api/sessions/{quote(session['id'], safe='')}"
        if args.operation == "close":
            result = request(origin, route + "/close", {})
        else:
            if args.scenario is None:
                raise ValueError("Task requires a scenario from the native catalog")
            catalog = request(origin, route + "/tasks")
            submission = {"scenario": args.scenario, "requestId": str(uuid4()),
                          "catalogRevision": catalog["descriptor"]["digest"]}
            if args.instruction is not None:
                submission["instruction"] = args.instruction.read_text(encoding="utf-8")
            result = request(origin, route + "/tasks", submission)
            if args.output is not None:
                if args.output.exists():
                    raise FileExistsError(args.output)
                save_json(args.output, result)
    if args.operation in ("open", "close"):
        result = {key: result[key] for key in ("id", "profileId", "state", "resources")}
    elif args.operation == "export":
        result = {key: result[key] for key in ("runId", "runState", "eventCount", "imageCount", "runError")}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
