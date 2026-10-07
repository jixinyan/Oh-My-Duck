import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import quote, urlsplit
from uuid import uuid4

from accept_navigation_replay import verify_navigation
from record_harness_demo import TERMINAL_STATES, export_run, request, save_json


def close_session(origin, session, output):
    route = "/api/sessions/" + quote(session["id"], safe="")
    closure = request(origin, route)
    if closure["state"] != "closed" or closure["resources"] != "released":
        closure = request(origin, route + "/close", {})
    save_json(output / "closure.json", closure)
    while closure["state"] != "closed" or closure["resources"] != "released":
        if closure["state"] == "error":
            raise RuntimeError("Native navigation session cleanup failed")
        time.sleep(10)
        closure = request(origin, route)
        save_json(output / "closure.json", closure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-distance-m", type=float, default=3.0)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment
            or parsed.username or parsed.password or parsed.path not in {"", "/"}):
        raise ValueError("Harness endpoint must be an HTTP(S) origin")
    if not math.isfinite(args.minimum_distance_m) or args.minimum_distance_m <= 0:
        raise ValueError("Minimum measured distance must be positive and finite")
    configuration = json.loads(args.scene_config.read_text())
    instruction = args.instruction.read_text()
    if not instruction.strip() or configuration["backend"] != "isaac-newton":
        raise ValueError("Navigation requires an instruction and a Newton scene")
    args.data_directory.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "scene.json", configuration)
    (args.output / "instruction.md").write_text(instruction)
    origin = args.base_url.rstrip("/")
    opening = {"profileId": configuration["scene_id"], "requestId": str(uuid4())}
    save_json(args.output / "opening.json", opening)
    session = None
    try:
        session = request(origin, "/api/sessions", opening)
        save_json(args.output / "session.json", session)
        route = "/api/sessions/" + quote(session["id"], safe="")
        catalog = request(origin, route + "/tasks")
        submission = {"scenario": "navigate-" + configuration["scene_id"], "requestId": str(uuid4()),
                      "catalogRevision": catalog["descriptor"]["digest"], "instruction": instruction}
        task = request(origin, route + "/tasks", submission)
        save_json(args.output / "task.json", task)
        run_route = "/api/runs/" + quote(task["runId"], safe="") + "?events=none"
        last = None
        with (args.output / "status.jsonl").open("x") as status:
            while True:
                run = request(origin, run_route)
                snapshot = {key: run[key] for key in ("id", "state", "updatedAt", "eventCount")}
                snapshot["executions"] = [{key: execution.get(key) for key in
                    ("execution_id", "state", "control_steps", "raw_sim_steps", "device_confirmed", "stop_reason")}
                    for execution in run["executions"]]
                if snapshot != last:
                    status.write(json.dumps(snapshot, allow_nan=False) + "\n")
                    status.flush()
                    last = snapshot
                if run["state"] in TERMINAL_STATES:
                    break
                time.sleep(10)
        close_session(origin, session, args.output)
        export_run(origin, task["runId"], args.output / "replay", args.data_directory)
        report = verify_navigation((args.output / "replay").resolve(strict=True), configuration, args.minimum_distance_m)
        report.update(scene_config_sha256=hashlib.sha256(args.scene_config.read_bytes()).hexdigest(),
                      instruction_sha256=hashlib.sha256(args.instruction.read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      auditor_sha256=hashlib.sha256(Path(__file__).with_name("accept_navigation_replay.py").read_bytes()).hexdigest())
        save_json(args.output / "physical-acceptance.json", report)
    finally:
        if session is None:
            identities = request(origin, "/api/sessions")["sessions"]
            records = [request(origin, "/api/sessions/" + quote(row["id"], safe="")) for row in identities]
            matches = [row for row in records if row["requestId"] == opening["requestId"]]
            if len(matches) > 1:
                raise RuntimeError("Native opening request has multiple session identities")
            if matches:
                session = matches[0]
                save_json(args.output / "session.json", session)
        if session is not None:
            close_session(origin, session, args.output)
    save_json(args.output / "result.json", {"status": "passed", "run_id": task["runId"],
              "session_id": session["id"], "resources_released": True,
              "physical_acceptance_sha256": hashlib.sha256((args.output / "physical-acceptance.json").read_bytes()).hexdigest()})
    print(json.dumps({"status": "passed", "run_id": task["runId"], "resources_released": True}))


if __name__ == "__main__":
    main()
