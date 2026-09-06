"""Process adapter for the isolated Isaac/Newton package; no simulator imports."""
from pathlib import Path
from typing import Sequence
from oh_my_duck.rl.training.base import BackendCommand

DIAGNOSTIC_TASK = "Omd-Microduck-PD-Diagnostic-v0"


class IsaacNewtonTrainingBackend:
    def __init__(self, root: Path):
        self.root = root

    def command(self, operation: str, arguments: Sequence[str]) -> BackendCommand:
        from oh_my_duck.rl.training.registry import BackendUnavailable
        interpreter = self.root / (".envs/isaac-assets/bin/python" if operation == "assets" else ".envs/isaac-newton/bin/python")
        if not interpreter.exists():
            raise BackendUnavailable("Run python omd.py setup --backend isaac-newton first; Newton is never replaced by PhysX")
        if operation not in {"assets", "probe", "train", "eval", "export"}:
            raise ValueError(f"Unsupported Isaac/Newton operation: {operation}")
        if operation == "train":
            tasks = [arg.split("=", 1)[1] for arg in arguments if arg.startswith("--task=")]
            for index, arg in enumerate(arguments):
                if arg == "--task" and index + 1 < len(arguments):
                    tasks.append(arguments[index + 1])
            if tasks != [DIAGNOSTIC_TASK]:
                raise BackendUnavailable("BAM locomotion training is not implemented. Only explicit --task " + DIAGNOSTIC_TASK + " is available for pipeline diagnostics.")
        return BackendCommand((str(interpreter), "-m", "oh_my_duck.rl.backends.isaac_newton." + operation, *arguments), self.root,
            {"PYTHONPATH": str(self.root / "src"),
             "ISAACLAB_PATH": str(self.root / ".cache/upstream/IsaacLab"),
             "OMD_PROJECT_ROOT": str(self.root), "MUJOCO_GL": "egl", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
