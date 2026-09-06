"""Launch pinned Isaac Lab SB3 PPO on the shared Newton diagnostic environment."""
import importlib.util
import os
import sys
from .paths import project_root
from .tasks import TASK_ID, register_tasks


def main():
    args = sys.argv[1:]
    tasks = [arg.split("=", 1)[1] for arg in args if arg.startswith("--task=")]
    tasks += [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--task"]
    if tasks != [TASK_ID]:
        raise SystemExit(f"Only explicit --task {TASK_ID} is implemented; this is not BAM locomotion")
    if any(arg.split("=", 1)[0] in {"--checkpoint", "--external_callback"} for arg in args):
        raise SystemExit("SB3 resume is pending normalization-state validation; custom registration is unsupported")
    if importlib.util.find_spec("stable_baselines3") is None:
        raise SystemExit("Install SB3 first: python omd.py setup --backend isaac-newton --rl-framework sb3")
    upstream = project_root() / ".cache/upstream/IsaacLab/scripts/reinforcement_learning"
    script = upstream / "sb3/train_sb3.py"
    register_tasks()
    os.chdir(project_root())
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("omd_upstream_sb3_train", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run([*args, "--headless"])


if __name__ == "__main__":
    main()
