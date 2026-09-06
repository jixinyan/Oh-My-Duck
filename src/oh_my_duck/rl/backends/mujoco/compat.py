# Modified by Oh My Duck: project-owned namespace, registration and module layout.
"""Pinned mjlab/RSL compatibility fixes; installed only by the backend plugin."""

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
from rsl_rl.algorithms.ppo import PPO as _PPO

# ---------------------------------------------------------------------------
# Patch 1: RewardManager.compute — sanitize NaN rewards before they enter the
# PPO buffer.  mjlab computes rewards BEFORE resetting environments, so any
# reward term operating on a NaN physics state returns NaN.  That NaN
# propagates: NaN reward → NaN advantage → NaN loss → NaN gradient →
# NaN/negative std → crash in torch.normal on the next mini-batch.
# ---------------------------------------------------------------------------
_orig_reward_compute = _RewardManager.compute

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums is updated inside compute() before nan_to_num can act.
    # Sanitize in-place so per-term metrics don't show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)

_RewardManager.compute = _nan_safe_reward_compute

# ---------------------------------------------------------------------------
# Patch 2: PPO.compute_returns — sanitize advantages before normalization.
# At a sudden curriculum step (e.g. reward weight ×2.5) the value function is
# badly wrong: all TD errors shift by the same amount, std(advantages) → tiny,
# and (A − mean) / (std + 1e-8) → huge.  That blows up the gradient for std,
# which the optimizer then pushes below zero.  Zeroing NaN/Inf advantages
# before normalization keeps them in a safe range.
# ---------------------------------------------------------------------------
_orig_compute_returns = _PPO.compute_returns

def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)

_PPO.compute_returns = _safe_compute_returns

# Patch 3 (ActorCritic._update_distribution std-clamp) was REMOVED in the mjlab
# 1.3.0 migration: rsl_rl 5.0.1 refactored the policy (no ActorCritic class; the
# distribution now lives in rsl_rl.modules.distribution). It was a defensive
# band-aid against std going negative/NaN (microban runs fine without it). If
# std-blowup recurs under 1.3.0, reinstate it against the new GaussianDistribution.



# ---------------------------------------------------------------------------
# Patch 4: exporter_utils.get_base_metadata — the new microduck model has
# passive joints (jaw linkage closed via equality constraints) that are part
# of the articulation but have no XML actuator.  The upstream exporter
# iterates robot.joint_names (16) and indexes joint_name_to_ctrl_id (14),
# crashing with KeyError on passive_*.  Filter passive joints out of the
# exported metadata so policies stay consistent with the 14-dim action space.
# ---------------------------------------------------------------------------
from mjlab.rl import exporter_utils as _exporter_utils  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction  # noqa: E402

def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }

_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
# Also patch the already-imported reference in the velocity task exporter.
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter  # noqa: E402
    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass



