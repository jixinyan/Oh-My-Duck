"""Minimal diagnostic task terms; no BAM or locomotion-reward equivalence claim."""
import torch
from oh_my_duck.rl.backends.isaac_newton.contracts import joint_indices


def as_torch(value):
    return getattr(value, "torch", value)


def observation(env):
    robot = env.scene["robot"]
    ids = list(joint_indices(robot.joint_names))
    commands = torch.zeros((env.num_envs, 13), device=env.device)
    return torch.cat((as_torch(robot.data.root_ang_vel_b), as_torch(robot.data.projected_gravity_b),
        (as_torch(robot.data.joint_pos) - as_torch(robot.data.default_joint_pos))[:, ids],
        as_torch(robot.data.joint_vel)[:, ids], env.action_manager.action, commands), dim=-1)


def upright(env):
    return (-as_torch(env.scene["robot"].data.projected_gravity_b)[:, 2]).clamp(0, 1)


def pose_error(env):
    robot = env.scene["robot"]
    return ((as_torch(robot.data.joint_pos) - as_torch(robot.data.default_joint_pos)) ** 2).sum(-1)


def fallen(env):
    robot = env.scene["robot"]
    return (as_torch(robot.data.root_pos_w)[:, 2] < 0.065) | (as_torch(robot.data.projected_gravity_b)[:, 2] > -0.5)
