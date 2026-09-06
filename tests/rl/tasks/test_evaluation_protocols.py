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


def test_walking_rejects_falling_and_bad_tracking():
    protocol=walking()
    trace={'height':np.full(100,.115),'tilt':np.zeros(100),'twist':np.zeros((100,3)), 'command':np.zeros((100,3))}
    assert protocol.score(trace,True)['success']
    trace['twist'][:,0]=1.
    assert not protocol.score(trace,True)['success']
    trace['twist'][:]=0.
    trace['height'][-1]=.04
    assert not protocol.score(trace,True)['success']
