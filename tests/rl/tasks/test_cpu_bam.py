"""Use real MuJoCo Jacobians to check BAM's external-load calculation."""
import mujoco
import numpy as np
from bam.mujoco import MujocoController
from oh_my_duck.rl.evaluation.rehearsal import infer_policy as ip
from oh_my_duck.robotics.microduck.actuators.cpu_bam import MicroduckMujocoController


def test_floating_robot_load_excludes_actual_dof_friction_and_preserves_motor():
    bam = ip.load_bam_model(200., 7.4, None)
    model, data, corrected, names = ip.load_mujoco_with_bam(ip.MICRODUCK_XML, bam, .005, .1, 6.)
    assert isinstance(corrected, MicroduckMujocoController)
    native = MujocoController(bam, names, model, data, vin_drop_gain=.1, vin_min=6.)
    data.qpos[:7] = [0, 0, .3, 1, 0, 0, 0]
    data.qpos[corrected.qpos_indexes] = ip.DEFAULT_POSE
    data.qvel[corrected.dof_indexes] = np.linspace(.1, .5, 14)
    model.dof_frictionloss[corrected.dof_indexes] = np.linspace(.05, .2, 14)
    corrected.reset(data.qpos)
    native.reset(data.qpos)
    corrected.q_target += .1
    native.q_target += .1
    mujoco.mj_forward(model, data)
    # Independently map forces with MuJoCo's constraint Jacobian, not efc_id.
    forces = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                      data.efc_force, 0.)
    projected = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, projected, forces)
    ids = corrected.dof_indexes
    expected_load = -data.qfrc_bias[ids] + data.qfrc_constraint[ids] - projected[ids]
    expected_loss, expected_damping = bam.compute_frictions(data.qfrc_actuator[ids], expected_load, data.qvel[ids])
    assert not np.array_equal(corrected.joint_indexes, ids)
    native.update()
    old_loss = model.dof_frictionloss[ids].copy()
    native_motor = data.ctrl.copy()
    corrected.update()
    np.testing.assert_allclose(model.dof_frictionloss[ids], expected_loss, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.dof_damping[ids], expected_damping, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(data.ctrl, native_motor)
    assert np.max(np.abs(old_loss - expected_loss)) > 1e-3
    assert bam.actuator.vin == 7.4


def test_controller_reset_restores_clock_targets_and_sag_after_time_rewinds():
    bam = ip.load_bam_model(200., 7.4, None)
    model, data, controller, _ = ip.load_mujoco_with_bam(ip.MICRODUCK_XML, bam, .005, .1, 6.)
    controller.last_ts = 8.
    controller._prev_motor_torque[:] = .5
    mujoco.mj_resetData(model, data)
    controller.reset(data.qpos)
    assert controller.last_ts == data.time == 0.
    np.testing.assert_array_equal(controller.q_target, data.qpos[controller.qpos_indexes])
    np.testing.assert_array_equal(controller._prev_motor_torque, 0.)
