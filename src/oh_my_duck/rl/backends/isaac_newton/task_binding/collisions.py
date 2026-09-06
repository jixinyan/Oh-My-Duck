"""Compile official walk/groundcontact masks into Newton shape filters and pairs."""

from functools import partial
from collections import defaultdict
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone


def reference_model(model):
    from oh_my_duck.robotics.microduck.microduck_constants import (
        MICRODUCK_WALK_ROBOT_CFG,
        MICRODUCK_STANDUP_ROBOT_CFG,
    )

    return (
        {"walk": MICRODUCK_WALK_ROBOT_CFG, "groundcontact": MICRODUCK_STANDUP_ROBOT_CFG}[model]
        .build()
        .compile()
    )


def source_geom(label, reference):
    parts = label.split("/")
    matches = []
    for geom in range(reference.ngeom):
        if not (reference.geom_contype[geom] or reference.geom_conaffinity[geom]):
            continue
        name = reference.geom(geom).name
        body = reference.body(int(reference.geom_bodyid[geom])).name
        mesh = reference.mesh(int(reference.geom_dataid[geom])).name
        if parts[-3] == body and parts[-2] == (name or mesh + "_1"):
            matches.append(geom)
    if len(matches) != 1:
        raise ValueError(f"Unmapped or ambiguous source collision mesh: {label}, matches={matches}")
    return matches[0]


def align_collisions(*args, model):
    from newton import ShapeFlags
    from isaaclab_newton.physics.newton_manager import NewtonManager

    reference = reference_model(model)
    builder = NewtonManager._builder
    worlds = defaultdict(list)
    ground = []
    for i, label in enumerate(builder.shape_label):
        if not (builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES):
            continue
        if "/ground/" in label:
            ground.append(i)
        else:
            worlds[builder.shape_world[i]].append((i, source_geom(label, reference)))
    if len(ground) != 1:
        raise ValueError("Expected a single flat plane")
    builder.shape_collision_group[ground[0]] = 0
    expected = sum(
        bool(reference.geom_contype[i] or reference.geom_conaffinity[i]) for i in range(reference.ngeom)
    )
    for world, shapes in worlds.items():
        if len(shapes) != expected:
            raise ValueError(f"World {world}: missing source collision shapes")
        for shape, geom in shapes:
            if reference.geom_contype[geom] & 1 or reference.geom_conaffinity[geom] & 1:
                friction = reference.geom_friction[geom]
                builder.add_custom_values(
                    **{
                        "mujoco:pair_world": world,
                        "mujoco:pair_geom1": shape,
                        "mujoco:pair_geom2": ground[0],
                        "mujoco:pair_condim": int(reference.geom_condim[geom]),
                        "mujoco:pair_friction": [
                            friction[0],
                            friction[0],
                            friction[1],
                            friction[2],
                            friction[2],
                        ],
                        "mujoco:pair_solref": reference.geom_solref[geom].tolist(),
                        "mujoco:pair_solimp": reference.geom_solimp[geom].tolist(),
                        "mujoco:pair_margin": float(reference.geom_margin[geom]),
                        "mujoco:pair_gap": float(reference.geom_gap[geom]),
                    }
                )
            else:
                builder.add_shape_collision_filter_pair(shape, ground[0])
        for i, (a, ga) in enumerate(shapes):
            for b, gb in shapes[i + 1 :]:
                if not (
                    (reference.geom_contype[ga] & reference.geom_conaffinity[gb])
                    or (reference.geom_contype[gb] & reference.geom_conaffinity[ga])
                ):
                    builder.add_shape_collision_filter_pair(a, b)


@clone
def spawn_official(prim_path, cfg, translation=None, orientation=None, *, model, **kwargs):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager

    root = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    reference = reference_model(model)
    while True:
        instances = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    seen = []
    for prim in Usd.PrimRange(root):
        if (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ):
            continue
        geom = source_geom(str(prim.GetPath()), reference)
        seen.append(geom)
        for name, value in [
            ("newton:maxHullVertices", -1),
            ("mjc:maxhullvert", -1),
            ("mjc:condim", int(reference.geom_condim[geom])),
            ("mjc:priority", int(reference.geom_priority[geom])),
        ]:
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Int).Set(value)
        material = UsdShade.Material.Define(stage, str(root.GetPath()) + f"/PhysicsMaterial{geom}")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(reference.geom_friction[geom, 0]))
        physics.CreateDynamicFrictionAttr(float(reference.geom_friction[geom, 0]))
        physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    expected = {
        i for i in range(reference.ngeom) if reference.geom_contype[i] or reference.geom_conaffinity[i]
    }
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError("USD collision set differs from source model")
    NewtonManager.register_callback(
        partial(align_collisions, model=model),
        PhysicsEvent.MODEL_INIT,
        name="omd_official_task_collision_groups",
        wrap_weak_ref=False,
    )
    return root


def configure_scene(scene_cfg, model):
    scene_cfg.robot.spawn.func = partial(spawn_official, model=model)
