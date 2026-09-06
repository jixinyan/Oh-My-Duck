"""Delegate diagnostic PPO training to the pinned Isaac Lab RSL-RL entry point."""
import os
from pathlib import Path
import sys
from .paths import project_root
from .tasks import TASK_ID


def main():
    args = sys.argv[1:]
    tasks = [arg.split("=", 1)[1] for arg in args if arg.startswith("--task=")]
    tasks += [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--task"]
    if tasks != [TASK_ID]:
        raise SystemExit(f"Only --task {TASK_ID} is implemented; this is not BAM locomotion training")
    if any(arg.startswith("--external_callback") for arg in args):
        raise SystemExit("External registration is owned by this backend")
    upstream = project_root() / ".cache/upstream/IsaacLab"
    script = upstream / "scripts/reinforcement_learning/rsl_rl/train.py"
    if not script.is_file():
        raise FileNotFoundError("Pinned Isaac Lab trainer missing; run setup")
    os.environ["PYTHONPATH"] = str(script.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.chdir(project_root())
    os.execv(sys.executable, [sys.executable, str(script), *args,
        "--external_callback", "omd_isaac.tasks.register_tasks", "--headless"])

if __name__ == "__main__":
    main()
