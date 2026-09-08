import fcntl
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from oh_my_duck.rl.experiments.gpu_pool import gpu_lease
from oh_my_duck.rl.experiments.adopt_preparation import free_devices
from oh_my_duck.rl.experiments.preparation import validate_preparation, REQUIRED


def test_stage_lease_reentrant_and_released_before_cpu_work(tmp_path):
    with patch.dict(os.environ, {'OMD_GPU_POOL': str(tmp_path), 'OMD_GPU_POOL_DEVICES': '3,5', 'CUDA_VISIBLE_DEVICES': '7'}):
        with gpu_lease() as first:
            assert first == '3'
            with gpu_lease() as nested:
                assert nested == first
            with (tmp_path/'3.lock').open('a') as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
        with (tmp_path/'3.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with gpu_lease() as second:
                assert second == '5'


def test_ready_training_uses_free_gpu_without_all_preparations_finishing():
    prep = {'a': {'device':'1','status':'completed'}, 'b': {'device':'3','status':'running'}}
    assert free_devices(['1','3'], prep, {}) == ['1']
    assert free_devices(['1','3'], prep, {'a': {'device':'1','status':'running'}}) == []
    prep['c'] = {'status':'queued','device':None}
    assert free_devices(['1','3'], prep, {}) == []


def test_partial_or_changed_preparation_cannot_bypass_gates(tmp_path):
    directory = tmp_path/'prepare'/'task';directory.mkdir(parents=True)
    path = directory/'result.json';spec={'id':'task','num_envs':8192}
    data={'status':'prepared','spec':spec,'stages':{name:{'status':'completed'} for name in REQUIRED}}
    path.write_text(json.dumps(data))
    (directory.parent/'campaign.json').write_text(json.dumps({'source_commit':'old'}))
    for name in ('smoke-export','capacity-export'):
        (directory/name).mkdir();(directory/name/'policy.onnx').write_bytes(b'fixture')
    with patch('oh_my_duck.rl.experiments.preparation.subprocess.check_output',return_value=''):
        assert validate_preparation(path,spec,tmp_path)['source_commit']=='old'
        with pytest.raises(ValueError,match='entire training spec'):
            validate_preparation(path,{**spec,'num_envs':16384},tmp_path)
        data['stages']['smoke-rehearsal']['status']='running';path.write_text(json.dumps(data))
        with pytest.raises(ValueError,match='required successful gate'):
            validate_preparation(path,spec,tmp_path)
    data['stages']['smoke-rehearsal']['status']='completed';path.write_text(json.dumps(data))
    with patch('oh_my_duck.rl.experiments.preparation.subprocess.check_output',return_value='src/oh_my_duck/rl/tasks/reward.py'):
        with pytest.raises(ValueError,match='inputs changed'):
            validate_preparation(path,spec,tmp_path)
