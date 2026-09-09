"""Behavior gates must reject reset shortcuts and transient target crossings."""
import numpy as np
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
