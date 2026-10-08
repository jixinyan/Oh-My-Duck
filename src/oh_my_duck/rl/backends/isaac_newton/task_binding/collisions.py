from functools import partial
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from oh_my_duck.rl.backends.isaac_newton.task_binding.collision_assets import (
    prepare_collision_assets, reference_model,
)
from oh_my_duck.rl.backends.isaac_newton.task_binding.collision_model import align_collision_model


def align_collisions(*args, model):
    from isaaclab_newton.physics.newton_manager import NewtonManager
    align_collision_model(NewtonManager._builder, model=model)


@clone
def spawn_official(prim_path, cfg, translation=None, orientation=None, *, model, **kwargs):
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager

    root = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    reference = reference_model(model)
    prepare_collision_assets(root, reference)
    NewtonManager.register_callback(
        partial(align_collisions, model=model),
        PhysicsEvent.MODEL_INIT,
        name="omd_official_task_collision_groups",
        wrap_weak_ref=False,
    )
    return root


def configure_scene(scene_cfg, model):
    scene_cfg.robot.spawn.func = partial(spawn_official, model=model)
