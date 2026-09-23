"""W&B lifecycle for native trainers that do not already own a W&B writer."""
import json
import os
from pathlib import Path
import subprocess

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def settings(root=None):
    repository = Path(root) if root is not None else project_root()
    configured = json.loads((repository / "configs/training.json").read_text())["logger"]
    mode = os.environ.get("WANDB_MODE", configured["mode"])
    if mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported W&B mode: {mode}")
    configured["mode"] = mode
    return configured


def start_run(*, backend, framework, task, directory=None, config=None):
    import wandb
    defaults = settings(ROOT)
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", defaults["project"]),
        entity=os.environ.get("WANDB_ENTITY") or defaults["entity"],
        mode=defaults["mode"],
        dir=str(directory or ROOT), sync_tensorboard=True,
        config={"backend": backend, "framework": framework, "task": task,
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "upstream": json.loads((ROOT / "configs/upstream.json").read_text()),
            **(config or {})})
    return run
