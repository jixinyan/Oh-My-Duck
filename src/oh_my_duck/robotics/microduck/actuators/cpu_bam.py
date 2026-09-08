"""CPU BAM correction for floating-base and passive-joint DOF indexing.

Pinned BAM 62bd8ce's CPU controller matches FRICTION_DOF rows by joint id.
MuJoCo stores a DOF id in those rows; its Warp BAM implementation already uses
that convention. Preserve the native motor, voltage sag and friction formulas,
then replace only the incorrectly indexed friction fields before physics steps.
"""
import mujoco
import numpy as np
from bam.mujoco import MujocoController


class MicroduckMujocoController(MujocoController):
    def reset(self, qpos):
        super().reset(qpos)
        # mj_resetData rewinds simulation time. Avoid a negative dt on the next
        # controller update (XL330's current proportional loop does not use dt).
        self.last_ts = self.mujoco_data.time

    def update(self):
        super().update()
        data, model = self.mujoco_data, self.mujoco_model
        friction = np.zeros(model.nv)
        mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        np.add.at(friction, data.efc_id[mask], data.efc_force[mask])
        ids = self.dof_indexes
        external = -data.qfrc_bias[ids] + data.qfrc_constraint[ids] - friction[ids]
        # compute_frictions is stateless. Native update's preliminary fields are
        # overwritten before mj_step; motor torque and sag history stay native.
        loss, damping = self.model.compute_frictions(
            data.qfrc_actuator[ids], external, data.qvel[ids]
        )
        model.dof_frictionloss[ids] = loss
        model.dof_damping[ids] = damping
