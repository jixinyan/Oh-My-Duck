"""Native mjlab discovery plugin; task ownership stays in the framework registry."""
from oh_my_duck.rl.training.tasks import project_tasks
from mjlab.tasks.registry import register_mjlab_task


from oh_my_duck.rl.tasks.recipes import build_environment


_registered = False


def register_tasks():
    global _registered
    if _registered:
        return
    from oh_my_duck.rl.backends.mujoco import compat  # activate compatibility only when binding this backend
    for task in project_tasks().list("mujoco"):
        binding = task.binding("mujoco")
        register_mjlab_task(task.id, build_environment(binding),
            build_environment(binding, play=True), binding.rsl_config.build(),
            binding.runner.resolve())

    _registered = True
