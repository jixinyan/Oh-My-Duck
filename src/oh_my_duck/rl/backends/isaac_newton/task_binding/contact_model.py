from oh_my_duck.rl.backends.isaac_newton.task_binding.collision_assets import source_geom


def configure_contact_model(actual, reference):
    import numpy as np

    for geom in range(actual.ngeom):
        label = actual.geom(geom).name
        if "/ground/" in label:
            # ground 使用包含官方材质参数的 explicit pair。
            actual.geom_contype[geom] = actual.geom_conaffinity[geom] = 0
            continue
        if label.startswith("/World/Environment/"):
            actual.geom_contype[geom] = actual.geom_conaffinity[geom] = 1
            continue
        ref = source_geom(label, reference)
        for name in (
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_solmix",
            "geom_condim",
            "geom_priority",
            "geom_margin",
            "geom_gap",
        ):
            getattr(actual, name)[geom] = getattr(reference, name)[ref]
    for name in ("contype", "conaffinity"):
        body = getattr(actual, "body_" + name)
        body[:] = 0
        np.bitwise_or.at(body, actual.geom_bodyid, getattr(actual, "geom_" + name))
