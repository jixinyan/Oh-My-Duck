"""W&B lifecycle for native trainers that do not already own a W&B writer."""
import json
import os
from pathlib import Path
import subprocess

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def settings():
    return json.loads((ROOT / "configs/training.json").read_text())["logger"]


def start_run(*, backend, framework, task, directory=None, config=None):
    import wandb
    defaults = settings()
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", defaults["project"]),
        entity=os.environ.get("WANDB_ENTITY") or defaults["entity"],
        mode=defaults["mode"],
        dir=str(directory or ROOT), sync_tensorboard=True,
        config={"backend": backend, "framework": framework, "task": task,
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "upstream": json.loads((ROOT / "configs/upstream.json").read_text()),
            **(config or {})})
    return run
