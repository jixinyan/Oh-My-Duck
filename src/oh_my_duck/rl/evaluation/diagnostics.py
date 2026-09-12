"""Explicit frozen-stage interventions and reward evidence for policy diagnosis."""
from copy import deepcopy
import math


def prepare_recipe(cfg, *, profile='standard', curriculum_step=None):
    if profile not in {'standard', 'training-stage'}:
        raise ValueError('Unknown evaluation profile')
    if (profile == 'training-stage') != (curriculum_step is not None):
        raise ValueError('training-stage requires an explicit curriculum step; standard forbids one')
    if curriculum_step is not None and (type(curriculum_step) is not int or curriculum_step < 0):
        raise ValueError('Curriculum step must be a nonnegative integer')
    stages = deepcopy(cfg.curriculum) if profile == 'training-stage' else {}
    cfg.curriculum = {}
    return stages


def apply_stage(env, stages, curriculum_step, push_scale):
    """Apply official managers once, then freeze before forced protocol resets.

    Keeping live curricula during protocol resets would overwrite the selected
    ground-pose probabilities and invalidate per-pose acceptance.
    """
    import torch
    from mjlab.managers.curriculum_manager import CurriculumManager
    if not math.isfinite(push_scale) or push_scale < 0:
        raise ValueError('Push scale must be finite and nonnegative')
    if stages:
        env.common_step_counter = curriculum_step
        manager = CurriculumManager(stages, env)
        manager.compute(torch.arange(env.num_envs, device=env.device))
    if 'push_robot' in env.cfg.events:
        term = env.event_manager.get_term_cfg('push_robot')
        term.params['velocity_range'] = {
            axis: tuple(v * push_scale for v in bounds)
            for axis, bounds in term.params['velocity_range'].items()
        }
    return {name: env.reward_manager.get_term_cfg(name).weight
            for name in env.reward_manager.active_terms}


def summarize_rewards(trace, names):
    import numpy as np
    rates = np.asarray(trace['weighted_reward_rate'])
    if rates.ndim != 2 or rates.shape[1] != len(names) or not np.isfinite(rates).all():
        raise ValueError('Invalid reward-rate trace')
    return {name: {'mean_rate': float(rates[:, i].mean()),
                   'final_second_mean_rate': float(rates[-50:, i].mean()),
                   'min_rate': float(rates[:, i].min()), 'max_rate': float(rates[:, i].max())}
            for i, name in enumerate(names)}
