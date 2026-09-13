import numpy as np
import pytest
from oh_my_duck.rl.evaluation.diagnostics import summarize_motion


def test_oscillation_is_distinguished_from_persistent_drift_without_rescoring():
    command = np.tile([.1, 0., 0.], (100, 1))
    twist = command.copy()
    twist[:, 1] = np.tile([-.2, .2], 50) + .03
    result = summarize_motion({'twist': twist, 'command': command})
    segment = result['segments'][0]
    assert result['diagnostic_only'] and 'success' not in result
    assert segment['instantaneous_rmse'][1] > .2
    assert segment['window_mean_rmse'][1] == pytest.approx(.03)
    assert segment['velocity_std'][1] == pytest.approx(.2)


def test_windows_do_not_cross_commands_and_stationary_policy_is_visible():
    command = np.r_[np.tile([.1, 0., 0.], (75, 1)), np.tile([0., 0., .5], (40, 1))]
    rows = summarize_motion({'twist': np.zeros_like(command), 'command': command})['segments']
    assert rows[0]['complete_windows'] == 1 and rows[0]['excluded_tail_ticks'] == 25
    assert rows[0]['window_mean_rmse'][0] == pytest.approx(.1)
    assert rows[1]['complete_windows'] == 0 and rows[1]['window_mean_rmse'] is None
    assert rows[1]['mean_twist'] == [0., 0., 0.]


def test_invalid_motion_is_rejected():
    with pytest.raises(ValueError):
        summarize_motion({'twist': [[np.nan, 0., 0.]], 'command': [[0., 0., 0.]]})
