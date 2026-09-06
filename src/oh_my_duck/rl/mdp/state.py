# Copyright 2026 Pollen Robotics. Licensed under Apache-2.0.
# Modified by Oh My Duck: split MDP responsibilities and make dependencies explicit.
"""Microduck state terms."""
import math
from dataclasses import dataclass as _dataclass
import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi, quat_apply, quat_from_angle_axis
from mjlab.rl import exporter_utils as _exporter_utils
from dataclasses import dataclass as _dataclass
from dataclasses import dataclass, field


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
