import pytest
from oh_my_duck.rl.learners.sb3.checkpoint import restore_progress


def test_restores_saved_curriculum_independently_of_environment_count():
    report = {'timesteps_after': 100000, 'num_envs': 64,
              'env_state': {'common_step_counter': 90000}}
    assert restore_progress(report, 100000) == (90000, 'saved_environment_state')


def test_legacy_fresh_run_counter():
    report = {'timesteps_before': 0, 'timesteps_after': 7680, 'num_envs': 64}
    assert restore_progress(report, 7680)[0] == 120


def test_legacy_resumed_run_recovers_actual_counter_and_reports_old_limitation():
    report = {'timesteps_before': 7680, 'timesteps_after': 15360, 'num_envs': 64}
    steps, source = restore_progress(report, 15360)
    assert steps == 120
    assert 'earlier legacy resumes restarted curricula' in source


def test_rejects_checkpoint_from_a_different_run():
    with pytest.raises(ValueError, match='does not match'):
        restore_progress({'timesteps_after': 7680}, 15360)


def test_rejects_invalid_saved_progress():
    with pytest.raises(ValueError, match='Invalid curriculum'):
        restore_progress({'timesteps_after': 7680, 'env_state': {'common_step_counter': -1}}, 7680)


def test_rejects_inconsistent_legacy_counts():
    with pytest.raises(ValueError, match='reconstruct'):
        restore_progress({'timesteps_before': 0, 'timesteps_after': 5, 'num_envs': 64}, 5)
