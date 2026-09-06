"""Delegate diagnostic PPO training to the pinned Isaac Lab RSL-RL entry point."""
import os
import runpy
from pathlib import Path
import sys
from oh_my_duck.rl.backends.isaac_newton.paths import project_root
from oh_my_duck.rl.backends.isaac_newton.tasks import TASK_ID, register_tasks


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
    # Upstream --help inspects the task before invoking external_callback.
    # Register in this process first, then run the unchanged official entry point.
    register_tasks()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args, "--external_callback", "oh_my_duck.rl.backends.isaac_newton.tasks.register_tasks", "--headless", "--visualizer", "none"]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__":
    main()
