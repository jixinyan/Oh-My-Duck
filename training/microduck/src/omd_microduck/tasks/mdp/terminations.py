# Copyright 2026 Pollen Robotics. Licensed under Apache-2.0.
# Modified by Oh My Duck: split MDP responsibilities and make dependencies explicit.
"""Microduck terminations terms."""
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
from .constants import _DEFAULT_ASSET_CFG


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """Terminate envs that have been continuously FALLEN for `max_duration_s`.

    For envs that mix walking with fall recovery (velstand): the fell_over
    termination gets disabled by curriculum so the policy can attempt recovery,
    but without a backstop a failed recovery farms recovery-reward for the whole
    20 s episode, starving the walk of data (audit: ~25% walking share). This
    gives every fall a fair recovery window, then recycles the env.
    """
    from .state import _fallen_mask
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # Freshly reset envs start with a clean timer.
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(
        fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s)
    )
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Terminate environments where MuJoCo produced NaN joint positions.

    MuJoCo's contact solver can overflow to NaN under extreme penetration or
    impulse (e.g. robot landing at high velocity). A NaN simulation state
    propagates into observations, corrupting the policy network weights.

    Terminating immediately resets the environment before the cascade spreads:
    - The observation returned to the runner is from the valid reset state.
    - NaN rewards are avoided on subsequent steps.

    Note: the reward at THIS terminal step may still be NaN from the simulation;
    mjlab computes rewards before resetting (see manager_based_rl_env.py step()).
    Our custom reward functions guard against NaN internally with nan_to_num,
    but standard mjlab rewards can still be NaN here. One NaN reward is
    tolerable because done=True prevents it propagating backward through GAE.

    Couvre TOUT l'état physique, pas seulement joint_pos : la divergence du
    contact fait souvent exploser le FREE-JOINT de base (position/orientation/
    vitesse) ou les ROUES passives, pas les joints actionnés. Ces quantités
    alimentent des termes d'obs critic (base_lin_vel, base_ang_vel,
    projected_gravity, wheel_vel) ; si on ne les surveille pas, l'env ne se
    reset pas et le NaN atteint l'obs → le check_nan de rsl_rl tue tout
    l'entraînement. On teste la non-finitude (NaN ET inf, l'inf devenant NaN en
    aval lors de la normalisation de projected_gravity).
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # Contact FORCES can blow up a step before qpos/qvel do: MuJoCo resolves a
    # degenerate contact into an inf/NaN impulse while the integrated state is
    # still finite. That force feeds the critic-only `foot_contact_forces` obs
    # (sign(F)*log1p(|F|)), which the state checks above do NOT cover — so the
    # env was not reset and the NaN reached the runner's check_nan, killing the
    # whole run (crash 2026-08-21, Velocity2-Rough-Backlash with hfield slopes).
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the trunk drops below ``min_height`` in world z.

    Utilisé par roller_slope comme « tombé dans le vide » : le terrain a un
    plat de sortie au bas de la rampe, donc une descente normale ne passe
    jamais sous le niveau du plat de sortie le plus bas. Choisir min_height
    en dessous de ce niveau => la terminaison ne se déclenche que si le robot
    quitte le solide et chute dans le vide. Indépendant de la géométrie exacte
    de la rampe (longueur/pente).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height
