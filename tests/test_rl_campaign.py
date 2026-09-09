import os
import unittest
from unittest.mock import patch
from oh_my_duck.rl.experiments.campaign import assigned_devices, worker_environment


class CampaignIsolation(unittest.TestCase):
    def test_allocation_order_preserved(self):
        self.assertEqual(assigned_devices('5,3,7,1', 2), ['5', '3'])

    def test_missing_duplicate_or_unsupported_devices_rejected(self):
        for devices,count in [('0',2),('0,0',2),('GPU-uuid',1)]:
            with self.assertRaises(ValueError):assigned_devices(devices,count)

    def test_workers_do_not_inherit_distributed_rank_or_online_logging(self):
        with patch.dict(os.environ, {'WORLD_SIZE':'8','RANK':'7','LOCAL_RANK':'7','WANDB_MODE':'online'}):
            env=worker_environment('3','test')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3')
        self.assertEqual(env['WANDB_MODE'],'offline')
        for key in ('WORLD_SIZE','RANK','LOCAL_RANK'):self.assertNotIn(key,env)

    def test_gpu_sharing_requires_explicit_capacity(self):
        self.assertEqual(assigned_devices('7', 6, runs_per_gpu=6), ['7'] * 6)
        self.assertEqual(assigned_devices('5,3', 3, runs_per_gpu=2), ['5', '5', '3'])
        with self.assertRaises(ValueError):
            assigned_devices('7', 6)
        with self.assertRaises(ValueError):
            assigned_devices('7', 1, runs_per_gpu=0)

    def test_queue_reuses_first_free_gpu_instead_of_waiting_for_slow_lane(self):
        import json
        from pathlib import Path
        import tempfile
        import threading
        from oh_my_duck.core.paths import project_root
        from oh_my_duck.rl.experiments import campaign
        plan = json.loads((project_root()/'configs/experiments/representative-full.json').read_text())
        plan['runs'] = plan['runs'][:3]
        third_started = threading.Event()
        observed = {}
        class Child:
            def __init__(self, command, **kwargs):
                self.name = command[command.index('--run-id')+1]
                self.pid = 1000 + len(observed)
                observed[self.name] = kwargs['env']['CUDA_VISIBLE_DEVICES']
                if self.name == plan['runs'][2]['id']:
                    third_started.set()
            def wait(self):
                if self.name == plan['runs'][0]['id']:
                    assert third_started.wait(2), 'Queued work waited for the busy GPU'
                return 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out'
            argv = ['campaign', '--config', 'unused.json', '--output', str(output), '--queue']
            config = Path(directory)/'config.json';config.write_text('{}')
            argv[2] = str(config)
            with patch('sys.argv', argv), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'1,6'}), \
                 patch.object(campaign, 'load_plan', return_value=plan), \
                 patch.object(campaign.subprocess, 'check_output', return_value='commit'), \
                 patch.object(campaign.subprocess, 'Popen', side_effect=Child), \
                 patch.object(campaign.signal, 'signal'):
                self.assertEqual(campaign.main(), 0)
            self.assertEqual(observed[plan['runs'][0]['id']], '1')
            self.assertEqual(observed[plan['runs'][1]['id']], '6')
            self.assertEqual(observed[plan['runs'][2]['id']], '6')
