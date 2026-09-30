from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a real cancelled EDH MicroDuck run")
    parser.add_argument("export_dir", type=Path)
    arguments = parser.parse_args()
    root = arguments.export_dir.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = json.loads((root / "source/events.json").read_text(encoding="utf-8"))
    if manifest["runState"] != "cancelled" or manifest["eventCount"] != len(events) or (
        [event["sequence"] for event in events] != list(range(1, len(events) + 1))
    ):
        raise AssertionError("Cancelled run identity or event order differs from the native export")
    paused = [event for event in events if event["type"] == "execution.updated" and
              event["detail"]["execution"]["state"] == "paused" and
              event["detail"]["execution"]["device_confirmed"]]
    cancelled = [event for event in events if event["type"] == "run.cancelled"]
    if len(paused) != 1 or len(cancelled) != 1 or cancelled[0]["sequence"] <= paused[0]["sequence"]:
        raise AssertionError("Cancellation lacks a confirmed physical stop boundary")
    boundary = paused[0]["detail"]["execution"]
    if boundary["control_steps"] <= 0 or boundary["raw_sim_steps"] != 4 * boundary["control_steps"]:
        raise AssertionError("Cancelled run lacks complete physical actions before stopping")
    if any(event["sequence"] > paused[0]["sequence"] and
           event["type"] in ("execution.updated", "simulation.frame") for event in events):
        raise AssertionError("Physical execution continued after confirmed cancellation boundary")
    print(json.dumps({"run_id": manifest["runId"], "run_state": manifest["runState"],
                      "events": len(events), "control_steps": boundary["control_steps"],
                      "raw_sim_steps": boundary["raw_sim_steps"],
                      "confirmed_boundary_event": paused[0]["sequence"],
                      "cancelled_event": cancelled[0]["sequence"]}, sort_keys=True))


if __name__ == "__main__":
    main()
