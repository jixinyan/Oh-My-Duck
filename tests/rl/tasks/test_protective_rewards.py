from types import SimpleNamespace

import torch

from oh_my_duck.rl.mdp.rewards_locomotion import (
    action_rate_l2_fallen_scaled,
    joint_torque_rate_l2_fallen_scaled,
    servo_acc_spike_penalty,
    servo_stall_penalty,
)


class _Scene:
    def __init__(self, asset):
        self.robot = asset
        self.terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))

    def __getitem__(self, name):
        return getattr(self, name)


def _env():
    data = SimpleNamespace(
        actuator_force=torch.tensor([[0.6, 0.0] * 7, [0.0, 0.0] * 7]),
        joint_vel=torch.zeros(2, 14),
        root_link_pos_w=torch.tensor([[0.0, 0.0, -0.10], [0.0, 0.0, 0.20]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    asset = SimpleNamespace(
        data=data,
        find_joints=lambda pattern: (list(range(14)), [f"servo_{i}" for i in range(14)]),
    )
    return SimpleNamespace(
        scene=_Scene(asset),
        device="cpu",
        num_envs=2,
        step_dt=0.01,
        episode_length_buf=torch.tensor([2, 2]),
        action_manager=SimpleNamespace(
            action=torch.ones(2, 14), prev_action=torch.zeros(2, 14)
        ),
    )


def test_servo_protection_terms_are_finite_nonnegative_and_servo_aligned():
    env = _env()
    stall = servo_stall_penalty(env)
    assert torch.equal(stall, torch.tensor([7.0, 0.0]))
    first = servo_acc_spike_penalty(env)
    env.scene.robot.data.joint_vel[0] = 4.0
    second = servo_acc_spike_penalty(env)
    assert torch.equal(first, torch.zeros(2))
    assert torch.all(second >= 0) and torch.isfinite(second).all()


def test_fallen_smoothness_terms_scale_only_fallen_envs():
    env = _env()
    action = action_rate_l2_fallen_scaled(env, fallen_scale=0.1, gate_tilt_above_deg=40)
    assert torch.allclose(action, torch.tensor([1.4, 14.0]))
    env.scene.robot.data.actuator_force.zero_()
    torque = joint_torque_rate_l2_fallen_scaled(env, fallen_scale=0.1)
    assert torch.isfinite(torque).all() and torch.all(torque >= 0)
