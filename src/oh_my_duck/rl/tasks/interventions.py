"""Opt-in training experiments; the official task factories remain the baseline."""
from copy import deepcopy
import math

WALKING_TASK = 'Mjlab-Velocity-Flat-MicroDuck'
LOW_SPEED_TRACKING_THRESHOLD = 0.2
LOW_SPEED_MINIMUM_COMMAND = 0.01
LOW_SPEED_TRACKING_BOOST_MAX = 4.0


def validate_action_rate_delay(task, iterations):
    if type(iterations) is not int or iterations < 0:
        raise ValueError('action_rate_delay_iterations must be a nonnegative integer')
    if iterations and task != WALKING_TASK:
        raise ValueError('Action-rate delay experiment is bound only to Flat Walking')


def apply_action_rate_delay(cfg, *, task, iterations=0, steps_per_iteration=24):
    """Delay positive-step smoothing stages only; preserve initial/final weights.

    This tests curriculum pacing, not a different PPO implementation or a reduced
    final task objective. Other command, DR and reward curricula are untouched.
    """
    validate_action_rate_delay(task, iterations)
    metadata = {'action_rate_delay_iterations': iterations}
    if not iterations:
        return metadata
    if steps_per_iteration != 24:
        raise ValueError('This experiment requires the official 24-step rollout')
    term = cfg.curriculum['action_rate_weight']
    if term.params['reward_name'] != 'action_rate_l2':
        raise ValueError('Unexpected action-rate curriculum binding')
    stages = term.params['weight_stages']
    if (not stages or stages[0] != {'step': 0, 'weight': -0.1}
            or cfg.rewards['action_rate_l2'].weight != -0.1
            or stages[-1] != {'step': 1500 * 24, 'weight': -1.0}):
        raise ValueError('Official smoothing recipe changed; review the experiment')
    term = deepcopy(term)
    for stage in term.params['weight_stages']:
        if stage['step'] > 0:
            stage['step'] += iterations * steps_per_iteration
    cfg.curriculum['action_rate_weight'] = term
    return metadata


def validate_low_speed_tracking_boost(task, boost):
    if isinstance(boost, bool) or not isinstance(boost, (int, float)):
        raise ValueError('low_speed_tracking_boost must be a finite non-negative number')
    if not math.isfinite(float(boost)) or boost < 0.0 or boost > LOW_SPEED_TRACKING_BOOST_MAX:
        raise ValueError(
            f'low_speed_tracking_boost must be between 0 and {LOW_SPEED_TRACKING_BOOST_MAX}'
        )
    if boost and task != WALKING_TASK:
        raise ValueError('Low-speed tracking experiment is bound only to Flat Walking')


def apply_low_speed_tracking_boost(cfg, *, task, boost=0.0):
    """Opt in to a bounded low-speed tracking signal for causal Walking tests."""
    validate_low_speed_tracking_boost(task, boost)
    metadata = {'low_speed_tracking_boost': float(boost)}
    if not boost:
        return metadata
    term = cfg.rewards.get('track_linear_velocity')
    if term is None or getattr(term.func, '__name__', '') != 'track_linear_velocity':
        raise ValueError('Official linear tracking reward binding changed; review the experiment')
    if term.params.get('command_name') != 'twist' or not math.isclose(
        float(term.params.get('std', 0.0)), math.sqrt(0.1), rel_tol=1e-7, abs_tol=1e-7
    ) or term.weight != 2.0:
        raise ValueError('Official linear tracking reward parameters changed; review the experiment')
    term = deepcopy(term)
    from oh_my_duck.rl import mdp as microduck_mdp
    term.func = microduck_mdp.track_linear_velocity_low_speed_boost
    term.params.update({
        'low_speed_threshold': LOW_SPEED_TRACKING_THRESHOLD,
        'minimum_speed': LOW_SPEED_MINIMUM_COMMAND,
        'boost': float(boost),
    })
    cfg.rewards['track_linear_velocity'] = term
    return metadata


def apply_training_interventions(
    cfg,
    *,
    task,
    action_rate_delay_iterations=0,
    low_speed_tracking_boost=0.0,
    steps_per_iteration=24,
):
    """Apply explicitly requested, resumable interventions and return identity."""
    metadata = apply_action_rate_delay(
        cfg, task=task, iterations=action_rate_delay_iterations,
        steps_per_iteration=steps_per_iteration,
    )
    metadata.update(apply_low_speed_tracking_boost(
        cfg, task=task, boost=low_speed_tracking_boost,
    ))
    return metadata


def validate_intervention_resume(previous, requested):
    defaults = {'action_rate_delay_iterations': 0, 'low_speed_tracking_boost': 0.0}
    saved = {**defaults, **previous.get('training_intervention', {})}
    wanted = {**defaults, **requested}
    if saved != wanted:
        raise ValueError('Resume cannot change the training intervention; start a fresh paired run')
