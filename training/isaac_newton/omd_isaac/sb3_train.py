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
    # The upstream launcher imports its wrapper inside run(). Replace that one
    # binding only for this process, retaining the upstream trainer unchanged.
    import isaaclab_rl.sb3 as sb3
    from .sb3_env import DiagnosticSb3VecEnvWrapper
    original = sb3.Sb3VecEnvWrapper
    sb3.Sb3VecEnvWrapper = DiagnosticSb3VecEnvWrapper
    tracking = None
    try:
        if not any(a in {"-h", "--help"} for a in args):
            sys.path.insert(0, str(project_root() / "training"))
            from common.tracking import start_run
            tracking = start_run(backend="isaac-newton", framework="sb3", task=TASK_ID,
                                 config={"native_arguments": args})
        module.run([*args, "--headless", "--visualizer", "none"])
    finally:
        sb3.Sb3VecEnvWrapper = original
        if tracking is not None:
            tracking.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == "__main__":
    main()
