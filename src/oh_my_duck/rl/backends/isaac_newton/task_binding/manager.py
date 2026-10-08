import mujoco_warp as mjw
import warp as wp
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from .collision_assets import reference_model
from .contact_model import configure_contact_model


class NewtonOfficialTaskManager(NewtonMJWarpManager):
    @classmethod
    def _build_solver(cls, model, solver_cfg):
        super()._build_solver(model, solver_cfg)
        if NewtonManager._graph is not None:
            raise RuntimeError("Official contact tables must be compiled before graph capture")
        solver = NewtonManager._solver
        actual, reference = solver.mj_model, reference_model(solver_cfg.robot_model)
        configure_contact_model(actual, reference)
        # 原生 put_model 编译接触表，solver 继续管理惯性、actuator 和状态数据。
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
