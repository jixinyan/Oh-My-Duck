"""Isaac actuator bridge to the unchanged official XL330 M6 BAM implementation.

Newton applies explicit joint efforts through qfrc_applied. The official BAM
implementation expects motor-side load in qfrc_actuator; the data view maps that
semantic difference without modifying BAM's torque, friction, sag or delay law.
"""
from types import SimpleNamespace
import numpy as np
import torch
import warp as wp
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils.configclass import configclass
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.sim.sim_data import WarpBridge
from .contracts import JOINT_NAMES
from .official import bam_cfg


class NewtonMotorData:
    def __init__(self, data):
        self.data = data

    @property
    def qfrc_actuator(self):
        return self.data.qfrc_actuator[...] + self.data.qfrc_applied[...]

    def __getattr__(self, name):
        return getattr(self.data, name)


class OfficialBamActuator(ActuatorBase):
    def __init__(self, cfg, *args, **kwargs):
        reference = bam_cfg().build(SimpleNamespace(), [], [])
        cfg.armature = float(reference._bam_model.actuator.get_extra_inertia())
        super().__init__(cfg, *args, **kwargs)
        if set(self.joint_names) != set(JOINT_NAMES):
            raise ValueError("BAM bridge currently supports the 14-servo walk model")
        self.official = None
        self.physics_calls = 0

    def _initialize_bam(self):
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton.solvers import SolverMuJoCo
        solver = NewtonManager._solver
        if not isinstance(solver, SolverMuJoCo) or solver.mjw_model is None:
            raise RuntimeError("Official BAM requires Newton GPU SolverMuJoCo")
        model = solver.mj_model
        names = [model.joint(i).name for i in range(model.njnt)]
        joint_ids = []
        for name in self.joint_names:
            matches = [i for i, full in enumerate(names) if full == name or full.endswith("/" + name) or full.endswith("_" + name)]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely map servo {name} into Newton joints: {names}")
            joint_ids.append(matches[0])
        entity = SimpleNamespace(joint_names=self.joint_names, indexing=SimpleNamespace(
            joint_ids=torch.tensor(joint_ids, device=self._device),
            ctrl_ids=torch.empty(0, dtype=torch.long, device=self._device)))
        config = bam_cfg()
        official = config.build(entity, list(range(self.num_joints)), self.joint_names)
        model_view = WarpBridge(solver.mjw_model, nworld=self._num_envs)
        data_view = NewtonMotorData(WarpBridge(solver.mjw_data, nworld=self._num_envs))
        # Newton's fields are already per-world. Refuse broadcasting: replacing
        # allocations after CUDA graph capture would silently leave stale pointers.
        for name in ("dof_frictionloss", "dof_damping", "dof_armature", "dof_solref", "dof_solimp"):
            field = getattr(model_view, name)
            if field.shape[0] != self._num_envs or (self._num_envs > 1 and field.stride(0) == 0):
                raise RuntimeError(f"Newton must expand {name} per world before BAM startup")
        ids = torch.tensor(model.jnt_dofadr[joint_ids], device=self._device)
        motor = official._bam_model.actuator
        self.force_limit = max(config.vin_range) * official._bam_model.kt.value / official._bam_model.R.value
        model_view.dof_frictionloss[:, ids] = 0.0
        model_view.dof_damping[:, ids] = 0.0
        model_view.dof_armature[:, ids] = motor.get_extra_inertia()
        if config.stiff_frictionloss:
            model_view.dof_solref[:, ids] = torch.tensor(official._STIFF_SOLREF_FRICTION, device=self._device)
            model_view.dof_solimp[:, ids] = torch.tensor(official._STIFF_SOLIMP_FRICTION, device=self._device)
        if not np.isclose(NewtonManager.get_solver_dt(), 0.005):
            raise ValueError("Official BAM delay and control law require 5 ms physics steps")
        # CPU template timestep is not necessarily updated until the first step.
        # BAM captures dt at initialization; use the actual Newton solver cadence.
        model.opt.timestep = NewtonManager.get_solver_dt()
        official.initialize(model, model_view, data_view, self._device)
        self.official = official
        self.solver = solver
        self.dof_ids = ids
        self.model_view = model_view

    def reset(self, env_ids=None):
        if self.official is not None:
            self.official.reset(env_ids)
            self.official.reset_friction_scale(slice(None) if env_ids is None else env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        if self.official is None:
            self._initialize_bam()
        cmd = ActuatorCmd(position_target=control_action.joint_positions,
            velocity_target=control_action.joint_velocities, effort_target=control_action.joint_efforts,
            pos=joint_pos, vel=joint_vel)
        self.computed_effort = self.official.compute(self.official.apply_delay(cmd))
        self.applied_effort = self.computed_effort.clamp(-self.force_limit, self.force_limit)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = control_action.joint_velocities = None
        self.physics_calls += 1
        return control_action


@configclass
class OfficialBamActuatorCfg(ActuatorBaseCfg):
    class_type = OfficialBamActuator
    stiffness = 0.0
    damping = 0.0
    friction = 0.0
    dynamic_friction = 0.0
    viscous_friction = 0.0
    # Resolved from the official fitted motor model in the actuator constructor.
    armature = None
