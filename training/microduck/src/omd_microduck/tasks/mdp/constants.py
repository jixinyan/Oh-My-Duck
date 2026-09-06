# Copyright 2026 Pollen Robotics. Licensed under Apache-2.0.
# Modified by Oh My Duck: split MDP responsibilities and make dependencies explicit.
"""Microduck constants terms."""
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


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))


_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))


_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


SPIN_PERIOD = 4.0


SPIN_RATE_MAX = 3.0


SPIN_ACCEL_END = 0.125


SPIN_HOLD_END = 0.525


SPIN_BRAKE_END = 0.650


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # atténuation du coût de dérive pendant le lancement


SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s ; recalibré sur la demi-voie mesurée et SPIN_RATE_MAX = 3.0


_ROULADE_FWD_SIGN = 1.0


_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"


_ROULADE_HEAD_SENSOR = "head_ground_contact"


_HEAD_LATCH_LO = math.radians(20.0)


_HEAD_LATCH_HI = math.radians(170.0)


_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)


_HEAD_TOP_DOWN_MIN = 0.3


_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below


_FLAT_ZERO = 0.866  # sin(60°): zero credit above
