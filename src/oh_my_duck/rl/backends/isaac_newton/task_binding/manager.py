"""Exact official contact tables before Newton captures its physics CUDA graph.

Newton 1.2.1 converts arbitrary shape filters through graph coloring. Coloring
preserves required collisions but can introduce extra pairs for mixed groups on
one body. Recompile collision tables from the owned source bitmasks at the native
manager's solver-construction extension point. No running graph is modified.
"""

import numpy as np
import mujoco_warp as mjw
import warp as wp
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from .collisions import reference_model, source_geom


class NewtonOfficialTaskManager(NewtonMJWarpManager):
    @classmethod
    def _build_solver(cls, model, solver_cfg):
        super()._build_solver(model, solver_cfg)
        if NewtonManager._graph is not None:
            raise RuntimeError("Official contact tables must be compiled before graph capture")
        solver = NewtonManager._solver
        actual, reference = solver.mj_model, reference_model(solver_cfg.robot_model)
        for geom in range(actual.ngeom):
            label = actual.geom(geom).name
            if "/ground/" in label:
                # Ground uses explicit pairs with the official robot material.
                actual.geom_contype[geom] = actual.geom_conaffinity[geom] = 0
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
        # Public put_model computes the same pair filtering as native MuJoCo Warp.
        # Only static collision tables are replaced; per-world inertial allocations,
        # actuator maps and Newton state buffers stay owned by the native solver.
        with wp.ScopedDevice(solver.device):
            compiled = mjw.put_model(actual)
        fields = (
            "geom_contype",
            "geom_conaffinity",
            "body_contype",
            "body_conaffinity",
            "nxn_geom_pair",
            "nxn_geom_pair_filtered",
            "nxn_pairid",
            "nxn_pairid_filtered",
            "geom_pair_type_count",
        )
        for name in fields:
            setattr(solver.mjw_model, name, getattr(compiled, name))
        for name in (
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_solmix",
            "geom_condim",
            "geom_priority",
            "geom_margin",
            "geom_gap",
        ):
            target = wp.to_torch(getattr(solver.mjw_model, name))
            source = wp.to_torch(getattr(compiled, name))
            target.copy_(source.expand_as(target))


@configclass
class OfficialTaskSolverCfg(MJWarpSolverCfg):
    class_type = "oh_my_duck.rl.backends.isaac_newton.task_binding.manager:NewtonOfficialTaskManager"
    robot_model: str = "walk"
