"""Preserve official walk collision groups and hulls in the Isaac USD import.

The official walk MJCF has two disjoint collision masks: feet/ground (1) and
leg/battery self-collision meshes (2). Newton expresses this with filter pairs.
"""
from collections import defaultdict
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone


def collider_group(label):
    if "/ground/" in label or "/left_foot_collision/" in label or "/right_foot_collision/" in label:
        return 1
    if "/power_support_1/" in label or "/leg_1/" in label:
        return 2
    raise ValueError(f"Unmapped walk collider: {label}")


def align_collision_groups(*_event_args):
    from newton import ShapeFlags
    from isaaclab_newton.physics.newton_manager import NewtonManager
    builder = NewtonManager._builder
    groups = defaultdict(lambda: {1: [], 2: []})
    ground = []
    for i, label in enumerate(builder.shape_label):
        if not (builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES):
            continue
        if "/ground/" in label:
            ground.append(i)
        else:
            groups[builder.shape_world[i]][collider_group(label)].append(i)
    if len(ground) != 1 or not groups:
        raise ValueError("Walk collision adapter requires one plane and replicated walk robots")
    # SolverMuJoCo cannot encode arbitrary filters against world-attached
    # planes through body exclusions. Its supported explicit-contact-pair API
    # preserves the plane and exactly the two official foot-ground contacts.
    from oh_my_duck.rl.backends.isaac_newton.official import bam_cfg
    bam_cfg()
    from oh_my_duck.robotics.microduck.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
    reference = MICRODUCK_WALK_ROBOT_CFG.build().compile()
    builder.shape_collision_group[ground[0]] = 0
    for world, group in groups.items():
        if len(group[1]) != 2 or len(group[2]) != 3:
            raise ValueError(f"World {world}: expected 2 foot and 3 self-only meshes, got {group}")
        for foot in group[1]:
            name = "left_foot_collision" if "/left_foot_collision/" in builder.shape_label[foot] else "right_foot_collision"
            geom = reference.geom(name).id
            friction = reference.geom_friction[geom]
            builder.add_custom_values(**{
                "mujoco:pair_world": world,
                "mujoco:pair_geom1": foot,
                "mujoco:pair_geom2": ground[0],
                "mujoco:pair_condim": int(reference.geom_condim[geom]),
                "mujoco:pair_friction": [friction[0], friction[0], friction[1], friction[2], friction[2]],
                "mujoco:pair_solref": reference.geom_solref[geom].tolist(),
                "mujoco:pair_solimp": reference.geom_solimp[geom].tolist(),
                "mujoco:pair_margin": float(reference.geom_margin[geom]),
                "mujoco:pair_gap": float(reference.geom_gap[geom]),
            })
        for self_only in group[2]:
            for other in ground + group[1]:
                builder.add_shape_collision_filter_pair(self_only, other)


@clone
def spawn_official_walk(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager
    root = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    # Author local overrides; cached source USD and upstream MJCF stay immutable.
    # De-instance before writing through referenced part geometry.
    while True:
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    material = UsdShade.Material.Define(stage, str(root.GetPath()) + "/BamPhysicsMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0)
    physics.CreateDynamicFrictionAttr(1.0)
    physics.CreateRestitutionAttr(0.0)
    colliders = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        group = collider_group(str(prim.GetPath()))
        colliders.append(prim)
        # Raw MuJoCo hulls have no 64-vertex approximation cap.
        prim.CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(-1)
        prim.CreateAttribute("mjc:maxhullvert", Sdf.ValueTypeNames.Int).Set(-1)
        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(3)
        prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1 if group == 1 else 0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    if len(colliders) != 5:
        raise ValueError(f"Expected 5 official walk collision meshes, found {len(colliders)}")
    NewtonManager.register_callback(align_collision_groups, PhysicsEvent.MODEL_INIT,
        name="omd_official_walk_collision_groups", wrap_weak_ref=False)
    return root


def configure_walk_scene(scene_cfg):
    scene_cfg.robot.spawn.func = spawn_official_walk
