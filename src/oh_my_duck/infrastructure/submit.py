#!/usr/bin/env python3
"""Submit an Alaya HTrain job with explicit resources and retained provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--project", default="agent")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= len(args.name) <= 33:
        parser.error("Provide a command after -- and a task name of 1–33 characters")
    source_root = ROOT
    if not args.dry_run:
        from oh_my_duck.infrastructure.snapshot import source_snapshot
        source_root, _ = source_snapshot(ROOT)
    command_text = ("cd " + shlex.quote(str(source_root)) + " && exec env "
        + shlex.join(["OMD_PROJECT_ROOT=" + str(source_root), "PYTHONPATH=" + str(source_root / "src"), *command]))
    log_dir = ROOT / "outputs/jobs" / args.name
    argv = ["submit", "--project", args.project, "--nodes", "1", "--gpus-per-node", str(args.gpus),
            "--name", args.name, "--log-path", str(log_dir / "scheduler.log"), "--cmd", command_text]
    print(shlex.join(argv), flush=True)
    if args.dry_run:
        return 0
    log_dir.mkdir(parents=True, exist_ok=False)
    def git(*parts):
        result = subprocess.run(["git", *parts], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    record = {"created_at": datetime.now(timezone.utc).isoformat(), "project": args.project,
              "nodes": 1, "gpus_per_node": args.gpus, "command": command,
              "cwd": str(source_root), "source_snapshot": str(source_root), "git_commit": git("rev-parse", "HEAD"),
              "git_status": git("status", "--porcelain"), "submit_argv": argv}
    (log_dir / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (log_dir / "submit-response.txt").write_text(result.stdout)
    print(result.stdout)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
