"""Explicit frozen-stage interventions and reward evidence for policy diagnosis."""
from copy import deepcopy
from dataclasses import replace
import math


def diagnostic_forward_protocol(protocol, speed=None):
    """Change only Walking's forward command, keeping the acceptance protocol intact."""
    if speed is None:
        return protocol
    if (protocol.kind != 'walking' or isinstance(speed, bool)
            or not isinstance(speed, (int, float)) or not math.isfinite(speed)
            or speed == 0 or abs(speed) > 0.4):
        raise ValueError('Diagnostic forward speed requires Walking and a nonzero speed within ±0.4 m/s')
    scenarios = tuple(replace(scenario, commands=tuple(
        (float(speed), command[1], command[2]) if command[0] != 0 else command
        for command in scenario.commands)) for scenario in protocol.scenarios)
    return replace(protocol, scenarios=scenarios)


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


def summarize_motion(trace, window_ticks=50):
    """Separate sustained tracking error from oscillation without rescoring acceptance.

    Windows stay inside constant-command segments, so stop/turn boundaries never
    dilute one another. Partial windows are excluded and their count is reported.
    """
    import numpy as np
    twist, command = np.asarray(trace['twist']), np.asarray(trace['command'])
    if (type(window_ticks) is not int or window_ticks < 1 or twist.ndim != 2
            or twist.shape[1] != 3 or twist.shape != command.shape or not len(twist)
            or not np.isfinite(twist).all() or not np.isfinite(command).all()):
        raise ValueError('Expected finite aligned Nx3 motion and command traces')
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(command, axis=0), axis=1)) + 1, len(command)]
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        velocity, requested = twist[start:end], command[start]
        count = len(velocity) // window_ticks
        windows = velocity[:count * window_ticks].reshape(count, window_ticks, 3).mean(1)
        segments.append({
            'start_tick': int(start), 'end_tick': int(end), 'command': requested.tolist(),
            'mean_twist': velocity.mean(0).tolist(),
            'instantaneous_rmse': np.sqrt(np.mean((velocity - requested) ** 2, axis=0)).tolist(),
            'velocity_std': velocity.std(0).tolist(),
            'window_mean_rmse': np.sqrt(np.mean((windows - requested) ** 2, axis=0)).tolist() if count else None,
            'complete_windows': count, 'excluded_tail_ticks': len(velocity) % window_ticks,
        })
    return {'diagnostic_only': True, 'window_ticks': window_ticks, 'segments': segments}
