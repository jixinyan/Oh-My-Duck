"""Process adapter for the isolated Isaac/Newton package; no simulator imports."""
from pathlib import Path
from typing import Sequence
from oh_my_duck.rl.training.base import BackendCommand

DIAGNOSTIC_TASK = "Omd-Microduck-PD-Diagnostic-v0"


def is_diagnostic(arguments):
    return DIAGNOSTIC_TASK in arguments or '--task='+DIAGNOSTIC_TASK in arguments


def require_task(root, arguments):
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.training.registry import BackendUnavailable
    if not arguments or arguments[0].startswith('-'):
        raise BackendUnavailable('Specify a registered task as the first argument, or explicit --task '+DIAGNOSTIC_TASK)
    try:
        return project_tasks(root).get(arguments[0]).binding('isaac-newton')
    except ValueError as error:
        raise BackendUnavailable(str(error)) from error


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
        if operation == 'eval' and '--task' in arguments:
            return BackendCommand((str(interpreter), '-m', 'oh_my_duck.rl.evaluation.task',
                *arguments, '--backend', 'isaac-newton'), self.root,
                {'PYTHONPATH': str(self.root/'src'), 'MUJOCO_GL': 'egl', 'MPLBACKEND': 'Agg'})
        if operation == 'train' and not is_diagnostic(arguments):
            require_task(self.root, arguments)
            return BackendCommand((str(interpreter), '-m', 'oh_my_duck.rl.learners.rsl_rl.train',
                *arguments, '--backend', 'isaac-newton'), self.root, self.environment())
        if operation == 'export' and arguments and not arguments[0].startswith('-'):
            require_task(self.root, arguments)
            # Export builds the official MuJoCo reference solely for metadata;
            # the learned actor and normalizer come from the explicit checkpoint.
            return BackendCommand((str(self.root / '.envs/mujoco/bin/python'), '-m', 'oh_my_duck.rl.artifacts.export', *arguments),
                self.root, self.environment())
        if operation == "train":
            tasks = [arg.split("=", 1)[1] for arg in arguments if arg.startswith("--task=")]
            for index, arg in enumerate(arguments):
                if arg == "--task" and index + 1 < len(arguments):
                    tasks.append(arguments[index + 1])
            if tasks != [DIAGNOSTIC_TASK]:
                raise BackendUnavailable("BAM locomotion training is not implemented. Only explicit --task " + DIAGNOSTIC_TASK + " is available for pipeline diagnostics.")
        return BackendCommand((str(interpreter), "-m", "oh_my_duck.rl.backends.isaac_newton." + operation, *arguments), self.root, self.environment())

    def environment(self):
        return {"PYTHONPATH": str(self.root / "src"),
             "ISAACLAB_PATH": str(self.root / ".cache/upstream/IsaacLab"),
             "OMD_PROJECT_ROOT": str(self.root), "MUJOCO_GL": "egl", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
