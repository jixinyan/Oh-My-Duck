"""Opt-in training experiments; the official task factories remain the baseline."""
from copy import deepcopy

WALKING_TASK = 'Mjlab-Velocity-Flat-MicroDuck'


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


def validate_intervention_resume(previous, requested):
    saved = previous.get('training_intervention', {'action_rate_delay_iterations': 0})
    if saved != requested:
        raise ValueError('Resume cannot change the training intervention; start a fresh paired run')
