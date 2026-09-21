import numpy as np
import pytest
from oh_my_duck.rl.evaluation.protocols import standup, walking


def test_standup_requires_a_sustained_final_pose_and_a_complete_episode():
    protocol=standup()
    trace={'height':np.full(100,.115),'tilt':np.zeros(100)}
    assert protocol.score(trace,True)['success']
    assert not protocol.score(trace,False)['success']
    trace['tilt'][-20]=1.
    assert not protocol.score(trace,True)['success']
    assert {s.name for s in protocol.scenarios} == {'standing','sitting','face_down','face_up'}
    for s in protocol.scenarios:
        assert sum(s.reset_probabilities.values()) == 1.
        assert s.reset_probabilities[s.name+'_prob'] == 1.


def walking_trace():
    protocol = walking()
    command = np.asarray(protocol.scenarios[0].commands)
    return protocol, {'height': np.full(len(command), .115), 'tilt': np.zeros(len(command)),
                      'twist': command.copy(), 'command': command}


def test_walking_rejects_falling_and_bad_tracking():
    protocol, trace = walking_trace()
    assert protocol.score(trace, True)['success']
    assert not protocol.score(trace, False)['success']
    trace['twist'][:, 0] = 1.
    assert not protocol.score(trace, True)['success']
    trace['twist'] = trace['command'].copy()
    trace['height'][-1] = .04
    assert not protocol.score(trace, True)['success']


def test_walking_rejects_stationary_policy_even_when_global_rmse_passes():
    protocol, trace = walking_trace()
    trace['twist'][:] = 0.
    result = protocol.score(trace, True)
    assert np.all(np.asarray(result['twist_rmse']) <= [.1, .1, .5])
    assert not result['success']
    assert [row['axis'] for row in result['command_response']] == ['vx', 'yaw_rate']
    assert all(row['response_fraction'] == 0. for row in result['command_response'])


def test_walking_requires_each_commanded_segment_and_correct_direction():
    protocol, trace = walking_trace()
    trace['twist'][:, 2] = 0.
    assert not protocol.score(trace, True)['success']
    trace['twist'] = -trace['command']
    assert not protocol.score(trace, True)['success']
    trace['twist'] = .75 * trace['command']
    assert protocol.score(trace, True)['success']


def test_walking_rejects_constant_motion_across_commands():
    protocol, trace = walking_trace()
    trace['twist'][:] = [.06, 0., .25]
    result = protocol.score(trace, True)
    assert np.all(np.asarray(result['twist_rmse']) <= [.1, .1, .5])
    assert not result['success']
    assert result['scoring_version'] == 3
    assert len(result['stages']) == 5
    assert all(not stage['success'] for stage in result['stages'])


@pytest.mark.parametrize('start,end,axis,value', [
    (300, 400, 0, .03), (600, 700, 2, .15),
    (100, 300, 2, .15), (400, 600, 0, .03),
    (100, 300, 1, .03), (100, 300, 0, .16),
])
def test_walking_checks_stop_cross_axis_motion_and_overspeed(start, end, axis, value):
    protocol, trace = walking_trace()
    trace['twist'][start:end, axis] = value
    result = protocol.score(trace, True)
    assert np.all(np.asarray(result['twist_rmse']) <= [.1, .1, .5])
    assert not result['success']


def test_walking_allows_transition_then_requires_sustained_response():
    protocol, trace = walking_trace()
    for start in (100, 300, 400, 600):
        trace['twist'][start:start + 25] = trace['command'][start - 1]
    result = protocol.score(trace, True)
    assert result['success']
    assert [stage['scoring_start_tick'] for stage in result['stages']] == [25, 125, 325, 425, 625]
    trace['twist'][200:225, 0] = 0.
    result = protocol.score(trace, True)
    assert result['command_response'][0]['response_fraction'] > .5
    assert not result['success']


def test_walking_stop_rms_rejects_canceling_oscillation():
    protocol, trace = walking_trace()
    trace['twist'][600:700, 0] = np.tile([.06, -.06], 50)
    result = protocol.score(trace, True)
    assert not result['success']
    assert result['stages'][-1]['window_rms_max'][0] == pytest.approx(.06)


def test_walking_scores_the_last_sample():
    protocol, trace = walking_trace()
    trace['twist'][-1, 2] = .6
    result = protocol.score(trace, True)
    assert not result['success']
    assert result['stages'][-1]['window_count'] == 51


def test_walking_rejects_a_segment_without_a_full_scoring_window():
    protocol, trace = walking_trace()
    trace = {key: values[:630] for key, values in trace.items()}
    result = protocol.score(trace, False)
    assert not result['success']
    assert not result['stages'][-1]['success']
    assert result['stages'][-1]['window_count'] == 0


@pytest.mark.parametrize('field', ['height', 'tilt', 'twist', 'command'])
def test_walking_rejects_nonfinite_trace(field):
    protocol, trace = walking_trace()
    trace[field][-1] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        protocol.score(trace, True)


def test_walking_rejects_inconsistent_trace_shapes():
    protocol, trace = walking_trace()
    trace['twist'] = trace['twist'][:-1]
    with pytest.raises(ValueError, match='matching nonempty'):
        protocol.score(trace, True)
