"""Task model views over the running Isaac/Newton solver; physics stays in Isaac.

The only private bridge is SolverMuJoCo's coordinate synchronization. It is pinned
and tested against Isaac articulation state, including writes before the first step.
"""

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from mjlab.sim.sim_data import WarpBridge
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.lab_api.math import matrix_from_quat
from oh_my_duck.rl.backends.isaac_newton.bam_actuator import NewtonMotorData


class NewtonSimulation:
    def __init__(self, native, cfg):
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton.solvers import SolverMuJoCo

        self.native, self.cfg = native, cfg
        self.manager = NewtonManager
        self.solver = NewtonManager._solver
        if not isinstance(self.solver, SolverMuJoCo) or self.solver.mjw_model is None:
            raise RuntimeError("A task binding requires the native Newton GPU solver")
        self.device = str(native.device)
        self.num_envs = native.scene.num_envs
        self.wp_device = wp.get_device(self.device)
        self.mj_model, self.wp_model, self.wp_data = (
            self.solver.mj_model,
            self.solver.mjw_model,
            self.solver.mjw_data,
        )
        self.model = WarpBridge(self.wp_model, nworld=self.num_envs)
        self.data = WarpBridge(self.wp_data)
        self.motor_data = NewtonMotorData(self.data)
        self._default_model_fields, self.expanded_fields = {}, set()
        self._reset_mask_wp = wp.zeros(self.num_envs, dtype=bool, device=self.device)
        self._reset_mask = wp.to_torch(self._reset_mask_wp)
        self.nan_guard = NanGuard(cfg.nan_guard, self.num_envs, self.mj_model)
        self.scene = None
        # Native ManagerBasedEnv must not fold decimation: shared task loop applies
        # the delayed BAM motor once on each of the four 5 ms physics steps.
        NewtonManager.set_decimation(1)
        self.forward()

    @property
    def default_model_fields(self):
        return self._default_model_fields

    def expand_model_fields(self, fields):
        for name in fields:
            field = getattr(self.model, name)
            if field.numel() and (
                field.shape[0] != self.num_envs or (self.num_envs > 1 and field.stride(0) == 0)
            ):
                raise RuntimeError(f"Newton field {name} is not independently allocated per world")
            self.get_default_field(name)
            self.expanded_fields.add(name)

    def get_default_field(self, name):
        if name not in self._default_model_fields:
            self._default_model_fields[name] = torch.as_tensor(
                np.array(getattr(self.mj_model, name)),
                device=self.device,
                dtype=getattr(self.model, name).dtype,
            ).clone()
        return self._default_model_fields[name]

    def recompute_constants(self, level):
        with wp.ScopedDevice(self.wp_device):
            getattr(mjw, level.name)(self.wp_model, self.wp_data)
        self.sync_model()

    def sync_model(self):
        """Keep Newton inertial state and explicit ground-pair friction consistent."""
        model = self.manager._model
        mapping = wp.to_torch(self.solver.mjc_body_to_newton).long()
        valid = mapping >= 0
        ids = mapping[valid]
        wp.to_torch(model.body_mass)[ids] = self.model.body_mass[valid]
        wp.to_torch(model.body_inv_mass)[ids] = self.model.body_mass[valid].reciprocal()
        wp.to_torch(model.body_com)[ids] = self.model.body_ipos[valid]
        rotation = matrix_from_quat(self.model.body_iquat[valid])
        inertia = rotation @ torch.diag_embed(self.model.body_inertia[valid]) @ rotation.transpose(-1, -2)
        wp.to_torch(model.body_inertia)[ids] = inertia
        wp.to_torch(model.body_inv_inertia)[ids] = torch.linalg.inv(inertia)
        dofs = wp.to_torch(self.solver.mjc_dof_to_newton_dof).long()
        valid_dofs = dofs >= 0
        wp.to_torch(model.joint_armature)[dofs[valid_dofs]] = self.model.dof_armature[valid_dofs]
        for pair in range(self.mj_model.npair):
            g1, g2 = self.mj_model.pair_geom1[pair], self.mj_model.pair_geom2[pair]
            robot_geom = g2 if self.mj_model.geom_bodyid[g1] == 0 else g1
            friction = self.model.geom_friction[:, robot_geom]
            self.model.pair_friction[:, pair] = friction[:, [0, 0, 1, 2, 2]]

    def forward(self):
        self.manager.forward()
        self.solver._update_mjc_data(self.wp_data, self.manager._model, self.manager._state_0)
        with wp.ScopedDevice(self.wp_device):
            mjw.forward(self.wp_model, self.wp_data)
            mjw.subtree_vel(self.wp_model, self.wp_data)
            mjw.rne_postconstraint(self.wp_model, self.wp_data)

    def step(self):
        with self.nan_guard.watch(self.data):
            self.native.scene.write_data_to_sim()
            self.native.sim.step(render=False)
            self.native.scene.update(self.cfg.mujoco.timestep)
        # These quantities would normally be requested by MJCF sensors. Newton's
        # USD import has no sensors, so explicitly compute the same derived state.
        with wp.ScopedDevice(self.wp_device):
            mjw.subtree_vel(self.wp_model, self.wp_data)
            mjw.rne_postconstraint(self.wp_model, self.wp_data)

    def reset(self, env_ids=None):
        self._reset_mask.zero_()
        self._reset_mask[slice(None) if env_ids is None else env_ids] = True
        with wp.ScopedDevice(self.wp_device):
            mjw.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)

    def sense(self):
        if self.scene is not None:
            for sensor in self.scene.sensors.values():
                sensor._invalidate_cache()
