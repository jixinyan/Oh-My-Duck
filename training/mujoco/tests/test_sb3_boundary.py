"""Use the real mjlab observation manager to catch stale-cache and delay regressions."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb3_env import TerminalObservationRecorder, MjlabSb3VecEnv
from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg


def state(env):
    return env.state


class TerminalTests(unittest.TestCase):
    def test_cached_observation_is_not_terminal_and_live_delay_is_unchanged(self):
        env = SimpleNamespace(num_envs=2, device="cpu", state=torch.ones(2, 61),
                              sim=SimpleNamespace(forward=lambda: None, sense=lambda: None))
        cfg = {"actor": ObservationGroupCfg(terms={"state": ObservationTermCfg(
            func=state, delay_min_lag=1, delay_max_lag=1)})}
        manager = env.observation_manager = ObservationManager(cfg, env)
        manager.compute(update_history=True)
        env.state.fill_(2)
        manager.compute(update_history=True)
        env.state.fill_(3)
        cached = manager.compute(update_history=False)["actor"].clone()
        self.assertEqual(cached[0, 0], 1)
        recorder = TerminalObservationRecorder(RecorderTermCfg(func=TerminalObservationRecorder), env)
        rng = torch.get_rng_state().clone()
        recorder.record_pre_reset(torch.tensor([1]))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(env._sb3_terminal[1][0, 0], 2)
        torch.testing.assert_close(manager.compute(update_history=False)["actor"], cached)
        # If the private computation advanced the live buffer this would be 3.
        self.assertEqual(manager.compute(update_history=True)["actor"][0, 0], 2)

    def test_timeout_and_termination_keep_separate_semantics(self):
        final = torch.full((2, 61), 7.)
        env = SimpleNamespace(num_envs=2, device="cpu")
        def step(actions):
            env._sb3_terminal = (torch.tensor([0, 1]), final)
            return {"actor": torch.zeros(2, 61)}, torch.ones(2), torch.tensor([False, True]), torch.tensor([True, True]), {}
        env.step = step
        wrapper = MjlabSb3VecEnv.__new__(MjlabSb3VecEnv)
        wrapper.env, wrapper.num_envs = env, 2
        wrapper.episode_returns, wrapper.episode_lengths = np.zeros(2), np.zeros(2)
        wrapper.terminal_count = wrapper.timeout_count = 0
        wrapper.step_async(np.zeros((2, 14)))
        obs, _, done, infos = wrapper.step_wait()
        self.assertTrue(done.all())
        self.assertTrue(infos[0]["TimeLimit.truncated"])
        self.assertFalse(infos[1]["TimeLimit.truncated"])
        self.assertEqual(infos[0]["terminal_observation"][0], 7.)
        self.assertEqual(obs[0, 0], 0.)
        final.zero_()
        self.assertEqual(infos[0]["terminal_observation"][0], 7.)


if __name__ == "__main__":
    unittest.main()
