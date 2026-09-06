"""Run one frozen ONNX through the same task battery in both simulation backends."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.training.registry import default_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--mujoco-renderer", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = project_root()
    reports = {}
    for backend in ("mujoco", "isaac-newton"):
        arguments = [
            "--task",
            args.task,
            "--policy",
            str(args.policy.resolve()),
            "--output",
            str((args.output / backend).resolve()),
            "--seed",
            str(args.seed),
            "--mujoco-renderer",
            args.mujoco_renderer,
        ]
        if args.video:
            arguments.append("--video")
        command = default_registry().get(backend, root).command("eval", arguments)
        with (args.output / (backend + ".log")).open("w") as log:
            code = subprocess.call(
                command.argv,
                cwd=command.cwd,
                env={**os.environ, **command.environment},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = args.output / backend / "result.json"
        reports[backend] = {
            "exit_code": code,
            "result": json.loads(path.read_text()) if path.exists() else None,
        }
    results = [item["result"] for item in reports.values()]
    valid = all(
        item["exit_code"] in (0, 2)
        and item["result"]
        and item["result"]["status"] in ("passed", "behavior_failed")
        for item in reports.values()
    )
    if valid:
        assert len({r["policy_sha256"] for r in results}) == 1
        assert all(r["task"] == args.task and r["seed"] == args.seed and not r["auto_reset"] for r in results)
        assert set(results[0]["scenarios"]) == set(results[1]["scenarios"])
    status = (
        "passed"
        if valid and all(r["status"] == "passed" for r in results)
        else ("behavior_failed" if valid else "error")
    )
    report = {
        "status": status,
        "task": args.task,
        "runs": reports,
        "scope": "Same policy and task battery; simulator dynamics need not be bit-identical",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if status == "passed" else (2 if status == "behavior_failed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
