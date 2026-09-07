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
