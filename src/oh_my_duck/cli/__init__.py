"""Public repository CLI; infrastructure runs without importing a training backend."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

COMMANDS = {
    "tasks": (None, "List official task inventory and the representative reproduction scope"),
    "frameworks": (None, "List RL frameworks, backend compatibility and validation status"),
    "status": (None, "Show component implementation status without initializing hardware"),
    "setup": ("bootstrap.py", "Fetch pinned sources/models and prepare a backend environment"),
    "campaign": ("campaign.py", "Run independent native RL pipelines, one per allocated GPU"),
    "submit": ("submit.py", "Submit a job with explicit resources and provenance"),
    "assets": ("run.py", "Build source-pinned assets for the selected backend"),
    "probe": ("run.py", "Check allocated GPU, physics and headless rendering"),
    "train": ("run.py", "Train using an explicitly selected backend"),
    "export": ("run.py", "Export through the selected backend"),
    "compare": ("run.py", "Run one policy through both simulation backends"),
    "rehearsal": ("run.py", "Run official CPU MuJoCo/BAM deployment batteries"),
    "package": ("run.py", "Create a local official policy package without uploading"),
    "eval": ("run.py", "Replay a policy headlessly and optionally save video"),
}


from oh_my_duck.core.paths import project_root


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: omd COMMAND [OPTIONS]  (or: python omd.py COMMAND)\n")
        for name, (_, description) in COMMANDS.items():
            print(f"  {name:8} {description}")
        print("\nUse COMMAND --help. Backend options follow --. See docs/architecture.md.")
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command {command!r}. Use --help.", file=sys.stderr)
        return 2
    root = project_root()
    if command == "tasks":
        parser = argparse.ArgumentParser(description=COMMANDS[command][1])
        parser.add_argument("--all", action="store_true", help="Include tasks outside the representative validation scope")
        args = parser.parse_args(sys.argv[2:])
        from oh_my_duck.rl.training.tasks import project_tasks
        catalog = project_tasks(root)
        selected = json.loads((root / "configs/training.json").read_text())["representative_tasks"]
        for task in catalog.list():
            if args.all or task.id in selected:
                print(task.id, "[representative]" if task.id in selected else "[inventory only]",
                      "backends=" + ",".join(task.bindings))
        print("Task registration does not imply training or behavior validation. See docs/rl-reproduction.md.")
        return 0
    if command == "frameworks":
        argparse.ArgumentParser(description=COMMANDS[command][1]).parse_args(sys.argv[2:])
        from oh_my_duck.rl.training.frameworks import default_framework_registry
        print(json.dumps(default_framework_registry().describe(), indent=2))
        return 0
    if command == "status":
        parser = argparse.ArgumentParser(description="Show declared component maturity, not robot capabilities.")
        parser.parse_args(sys.argv[2:])
        print((root / "configs/project.json").read_text())
        return 0
    filename = COMMANDS[command][0]
    module = importlib.import_module("oh_my_duck.rl.experiments.campaign" if command == "campaign"
                                    else "oh_my_duck.infrastructure." + filename.removesuffix(".py"))
    sys.argv = [sys.argv[0], *([command] if filename == "run.py" else []), *sys.argv[2:]]
    return module.main()
