from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from oh_my_duck.rl.tasks.interventions import (
    WALKING_TASK, apply_action_rate_delay, validate_action_rate_delay, validate_intervention_resume,
)
from oh_my_duck.rl.tasks.walking.environment import make_microduck_velocity_env_cfg
from oh_my_duck.rl.experiments.baseline_probe import describe
from oh_my_duck.rl.mdp.curricula import reward_weight


def test_zero_delay_is_an_exact_official_recipe_noop():
    cfg = make_microduck_velocity_env_cfg(play=False)
    before = describe(cfg)
    assert apply_action_rate_delay(cfg, task=WALKING_TASK) == {'action_rate_delay_iterations': 0}
    assert describe(cfg) == before


def test_only_positive_smoothing_stage_times_change_and_other_recipes_stay_pristine():
    cfg = make_microduck_velocity_env_cfg(play=False)
    baseline = deepcopy(cfg)
    apply_action_rate_delay(cfg, task=WALKING_TASK, iterations=4500)
    stages = cfg.curriculum['action_rate_weight'].params['weight_stages']
    assert [x['step'] // 24 for x in stages] == [0, 5000, 5250, 5500, 5750, 6000]
    assert [x['weight'] for x in stages] == [-.1, -.2, -.4, -.6, -.8, -1.]
    cfg.curriculum['action_rate_weight'] = baseline.curriculum['action_rate_weight']
    assert describe(cfg) == describe(baseline)
    fresh = make_microduck_velocity_env_cfg(play=False)
    assert describe(fresh) == describe(baseline)


def test_actual_live_weight_manager_obeys_delayed_boundary_and_restores_final_objective():
    cfg = make_microduck_velocity_env_cfg(play=False)
    apply_action_rate_delay(cfg, task=WALKING_TASK, iterations=4500)
    term = SimpleNamespace(weight=-.1)
    env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda name: term))
    for step, expected in [(1500*24+1,-.1),(5000*24,-.1),(5000*24+1,-.2),(6000*24+1,-1.)]:
        env.common_step_counter = step
        reward_weight(env, torch.arange(1), **cfg.curriculum['action_rate_weight'].params)
        assert term.weight == expected


@pytest.mark.parametrize('value', [-1, True, 1.5, '4500'])
def test_invalid_interventions_fail(value):
    with pytest.raises(ValueError): validate_action_rate_delay(WALKING_TASK, value)


def test_scope_and_resume_identity_cannot_silently_change():
    with pytest.raises(ValueError): validate_action_rate_delay('Mjlab-StandUp-Flat-MicroDuck',4500)
    validate_intervention_resume({}, {'action_rate_delay_iterations':0})
    validate_intervention_resume({'training_intervention':{'action_rate_delay_iterations':4500}}, {'action_rate_delay_iterations':4500})
    with pytest.raises(ValueError): validate_intervention_resume({}, {'action_rate_delay_iterations':4500})
    with pytest.raises(ValueError): validate_intervention_resume({'training_intervention':{'action_rate_delay_iterations':4500}}, {'action_rate_delay_iterations':0})


def test_full_campaign_has_four_matched_pairs_and_no_short_training_budget():
    from pathlib import Path
    from oh_my_duck.rl.experiments.campaign import load_plan
    from oh_my_duck.core.paths import project_root
    plan = load_plan(project_root() / 'configs/experiments/walking-smoothing-delay.json')
    assert len(plan['runs']) == 8 and plan['record_previews'] and plan['unforced_evaluation']
    for index in range(0, 8, 2):
        control, delayed = deepcopy(plan['runs'][index:index+2])
        assert control.pop('action_rate_delay_iterations') == 0
        assert delayed.pop('action_rate_delay_iterations') == 4500
        control.pop('id'); delayed.pop('id')
        assert control == delayed
        assert control['num_envs'] == 8192 and control['iterations'] == 50000
        assert control['seed'] == 42


def test_low_speed_tracking_boost_is_explicit_and_preserves_official_zero_case():
    from oh_my_duck.rl.tasks.interventions import apply_training_interventions

    cfg = make_microduck_velocity_env_cfg(play=False)
    baseline = describe(cfg)
    assert apply_training_interventions(cfg, task=WALKING_TASK) == {
        'action_rate_delay_iterations': 0,
        'low_speed_tracking_boost': 0.0,
    }
    assert describe(cfg) == baseline

    cfg = make_microduck_velocity_env_cfg(play=False)
    metadata = apply_training_interventions(cfg, task=WALKING_TASK, low_speed_tracking_boost=1.0)
    term = cfg.rewards['track_linear_velocity']
    assert metadata['low_speed_tracking_boost'] == 1.0
    assert term.func.__name__ == 'track_linear_velocity_low_speed_boost'
    assert term.params['low_speed_threshold'] == 0.2
    assert term.params['minimum_speed'] == 0.01
    assert term.params['boost'] == 1.0


def test_low_speed_intervention_rejects_other_tasks_and_bad_values():
    from oh_my_duck.rl.tasks.interventions import validate_low_speed_tracking_boost

    with pytest.raises(ValueError):
        validate_low_speed_tracking_boost('Mjlab-StandUp-Flat-MicroDuck', 1.0)
    for value in (-0.1, 4.1, float('inf'), True):
        with pytest.raises(ValueError):
            validate_low_speed_tracking_boost(WALKING_TASK, value)


def test_low_speed_intervention_resume_defaults_are_backward_compatible():
    from oh_my_duck.rl.tasks.interventions import validate_intervention_resume

    validate_intervention_resume(
        {'training_intervention': {'action_rate_delay_iterations': 0}},
        {'action_rate_delay_iterations': 0, 'low_speed_tracking_boost': 0.0},
    )
    with pytest.raises(ValueError):
        validate_intervention_resume(
            {'training_intervention': {'low_speed_tracking_boost': 1.0}},
            {'action_rate_delay_iterations': 0, 'low_speed_tracking_boost': 0.0},
        )


def test_low_speed_campaign_has_eight_matched_full_budget_runs():
    from oh_my_duck.rl.experiments.campaign import load_plan
    from oh_my_duck.core.paths import project_root

    plan = load_plan(project_root() / 'configs/experiments/walking-low-speed-boost.json')
    assert len(plan['runs']) == 8 and plan['record_previews'] and plan['unforced_evaluation']
    for index in range(0, 8, 2):
        control, boosted = deepcopy(plan['runs'][index:index + 2])
        assert control.pop('low_speed_tracking_boost') == 0.0
        assert boosted.pop('low_speed_tracking_boost') == 1.0
        control.pop('id')
        boosted.pop('id')
        assert control == boosted
        assert control['num_envs'] == 8192 and control['iterations'] == 50000
