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
    from oh_my_duck.rl.backends.isaac_newton.asset_names import get_isaac_allcollisions_cfg

    return (
        {"walk": MICRODUCK_WALK_ROBOT_CFG, "groundcontact": MICRODUCK_STANDUP_ROBOT_CFG,
         "allcollisions": get_isaac_allcollisions_cfg()}[model]
        .build()
        .compile()
    )


def source_geom(label, reference):
    parts = label.split("/")
    joint_names = {reference.joint(i).name for i in range(reference.njnt)}
    matches = []
    mesh_matches = []
    for geom in range(reference.ngeom):
        if not (reference.geom_contype[geom] or reference.geom_conaffinity[geom]):
            continue
        name = reference.geom(geom).name
        body = reference.body(int(reference.geom_bodyid[geom])).name
        body = body + "_1" if body in joint_names else body
        mesh = reference.mesh(int(reference.geom_dataid[geom])).name
        if parts[-3] == body and parts[-2] == (name or mesh + "_1"):
            matches.append(geom)
        if parts[-3] == body and parts[-1] == mesh:
            mesh_matches.append(geom)
    if not matches and len(mesh_matches) == 1:
        matches = mesh_matches
    if len(matches) != 1:
        raise ValueError(f"Unmapped or ambiguous source collision mesh: {label}, matches={matches}")
    return matches[0]


def align_collisions(*args, model):
    from newton import ShapeFlags, GeoType
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    reference = reference_model(model)
    builder = NewtonManager._builder
    builder.omd_scene_transform_audit = []
    for i, label in enumerate(builder.shape_label):
        scale = np.asarray(builder.shape_scale[i], dtype=float)
        if not label.startswith("/World/Environment/") or not np.any(scale < 0):
            continue
        kind = builder.shape_type[i]
        if kind in (GeoType.MESH, GeoType.CONVEX_MESH):
            mesh = builder.shape_source[i]
            signs = np.sign(scale)
            indices = mesh.indices.reshape(-1, 3).copy()
            if np.prod(signs) < 0:
                indices = indices[:, [0, 2, 1]]
            reflected = mesh.copy(vertices=mesh.vertices * signs, indices=indices.ravel())
            if reflected.normals is not None:
                reflected.normals[:] *= signs
            np.testing.assert_allclose(reflected.vertices * np.abs(scale), mesh.vertices * scale,
                                       atol=1e-7, rtol=1e-7)
            transform = builder.shape_transform[i]
            body = builder.shape_body[i]
            if body >= 0:
                transform = wp.transform_multiply(builder.body_q[body], transform)
            before = np.asarray([wp.transform_point(transform, wp.vec3(*v))
                                 for v in mesh.vertices * scale])
            after = np.asarray([wp.transform_point(transform, wp.vec3(*v))
                                for v in reflected.vertices * np.abs(scale)])
            bbox_before = [before.min(axis=0).tolist(), before.max(axis=0).tolist()]
            bbox_after = [after.min(axis=0).tolist(), after.max(axis=0).tolist()]
            builder.shape_source[i] = reflected
        elif kind not in (GeoType.BOX, GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.PLANE):
            raise ValueError(f"Unsupported signed-scale shape: {label}, type={kind}, scale={scale}")
        builder.shape_scale[i] = tuple(np.abs(scale))
        builder.omd_scene_transform_audit.append({"shape": label, "type": int(kind),
                                                 "source_scale": scale.tolist(),
                                                 "world_bbox_before": bbox_before if kind in (GeoType.MESH, GeoType.CONVEX_MESH) else None,
                                                 "world_bbox_after": bbox_after if kind in (GeoType.MESH, GeoType.CONVEX_MESH) else None,
                                                 "geometry_preserved": True})
    invalid = [{"shape": label, "type": int(builder.shape_type[i]),
                "scale": list(builder.shape_scale[i]), "flags": int(builder.shape_flags[i])}
               for i, label in enumerate(builder.shape_label)
               if builder.shape_type[i] not in (GeoType.PLANE, GeoType.NONE)
               and not np.any(np.asarray(builder.shape_scale[i]) > 0)]
    if invalid:
        raise ValueError(f"USD contains zero-size Newton shapes: {invalid}")
    worlds = defaultdict(list)
    ground = []
    environment = []
    for i, label in enumerate(builder.shape_label):
        if not (builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES):
            continue
        if "/ground/" in label:
            ground.append(i)
        elif label.startswith("/World/Environment/"):
            environment.append(i)
        else:
            worlds[builder.shape_world[i]].append((i, source_geom(label, reference)))
    if len(ground) > 1 or not ground and not environment:
        raise ValueError("Expected a flat plane or an external collision scene")
    for shape in ground + environment:
        builder.shape_collision_group[shape] = 0
    expected = sum(
        bool(reference.geom_contype[i] or reference.geom_conaffinity[i]) for i in range(reference.ngeom)
    )
    for world, shapes in worlds.items():
        if len(shapes) != expected:
            raise ValueError(f"World {world}: missing source collision shapes")
        for shape, geom in shapes:
            if ground and (reference.geom_contype[geom] & 1 or reference.geom_conaffinity[geom] & 1):
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
            elif ground:
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
