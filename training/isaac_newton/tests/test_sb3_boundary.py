"""CPU regression checks for SB3 bootstrapping, without creating a simulation."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omd_isaac.environment import DiagnosticEnv
from isaaclab.envs import ManagerBasedRLEnv


class TerminalCaptureTests(unittest.TestCase):
    def test_snapshot_precedes_reset_and_owns_its_storage(self):
        env = DiagnosticEnv.__new__(DiagnosticEnv)
        env._is_closed = True
        env._capture_terminal_observations = True
        state = torch.arange(3 * 61, dtype=torch.float32).reshape(3, 61)
        expected = state[[0, 2]].clone()
        with patch.object(ManagerBasedRLEnv, '_reset_idx', side_effect=lambda ids: state.zero_()):
            env.observation_manager = SimpleNamespace(compute=lambda **kwargs: {'policy': state})
            env._reset_idx(torch.tensor([0, 2]))
        ids, snapshot = env._terminal_snapshot
        torch.testing.assert_close(ids, torch.tensor([0, 2]))
        torch.testing.assert_close(snapshot, expected)
        self.assertTrue((state == 0).all())


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'optional SB3 extra not installed')
class Sb3BoundaryTests(unittest.TestCase):
    def wrapper(self, snapshot):
        from omd_isaac.sb3_env import DiagnosticSb3VecEnvWrapper
        wrapper = DiagnosticSb3VecEnvWrapper.__new__(DiagnosticSb3VecEnvWrapper)
        wrapper.env = SimpleNamespace(unwrapped=SimpleNamespace(_terminal_snapshot=snapshot))
        wrapper.num_envs = 3
        wrapper.fast_variant = True
        wrapper._ep_rew_buf = np.ones(3)
        wrapper._ep_len_buf = np.ones(3)
        wrapper._terminal_count = wrapper._timeout_count = 0
        return wrapper

    def test_timeout_bootstraps_from_final_state_not_reset_state(self):
        states = torch.stack([torch.ones(61), torch.full((61,), 9.0)])
        wrapper = self.wrapper((torch.tensor([0, 2]), states))
        infos = wrapper._process_extras(np.zeros((3, 61)), np.array([True, False, False]),
            np.array([False, False, True]), {}, np.array([0, 2]))
        self.assertFalse(infos[0]['TimeLimit.truncated'])
        self.assertTrue(infos[2]['TimeLimit.truncated'])
        np.testing.assert_array_equal(infos[2]['terminal_observation'], np.full(61, 9.0))
        states.zero_()
        self.assertEqual(infos[2]['terminal_observation'][0], 9.0)
        self.assertEqual(wrapper._terminal_count, 2)
        self.assertEqual(wrapper._timeout_count, 1)

    def test_missing_final_state_is_rejected(self):
        wrapper = self.wrapper(None)
        with self.assertRaisesRegex(RuntimeError, 'pre-reset'):
            wrapper._process_extras(np.zeros((3, 61)), np.zeros(3, dtype=bool),
                np.array([True, False, False]), {}, np.array([0]))

if __name__ == '__main__':
    unittest.main()
