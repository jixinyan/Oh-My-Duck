"""Native mjlab discovery plugin; task ownership stays in the framework registry."""
from oh_my_duck.rl.training.tasks import project_tasks
from mjlab.tasks.registry import register_mjlab_task


def build_environment(binding, *, play=False):
    cfg = binding.environment.build(play=play)
    if binding.robot_variant is not None:
        from oh_my_duck.rl.tasks.backlash import make_backlash_variant
        cfg = make_backlash_variant(cfg, binding.robot_variant.build())
    return cfg


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
