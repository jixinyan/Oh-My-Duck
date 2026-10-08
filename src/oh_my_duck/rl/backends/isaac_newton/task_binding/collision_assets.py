from pathlib import PurePosixPath


def reference_model(model):
    from oh_my_duck.robotics.microduck.microduck_constants import (
        MICRODUCK_WALK_ROBOT_CFG,
        MICRODUCK_STANDUP_ROBOT_CFG,
    )
    from oh_my_duck.rl.backends.isaac_newton.asset_names import get_isaac_allcollisions_cfg, get_isaac_rollers_cfg

    return (
        {"walk": MICRODUCK_WALK_ROBOT_CFG, "groundcontact": MICRODUCK_STANDUP_ROBOT_CFG,
         "allcollisions": get_isaac_allcollisions_cfg(),
         "groundcontact_rollers": get_isaac_rollers_cfg()}[model]
        .build()
        .compile()
    )


def source_geom(label, reference):
    parts = PurePosixPath(label).parts
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


def prepare_collision_assets(root, reference):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    stage = root.GetStage()
    expanded = 0
    while True:
        instances = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not instances:
            break
        expanded += len(instances)
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
    return {"collision_geoms": seen, "instances_expanded": expanded}
