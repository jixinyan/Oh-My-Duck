"""Native SB3 normalization and actor parity, including clipped outliers."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb3_export import NormalizedSB3Actor


class ContractEnv(Env):
    observation_space = Box(-np.inf, np.inf, (61,), np.float32)
    action_space = Box(-1000, 1000, (14,), np.float32)


class ExportTests(unittest.TestCase):
    def test_normalized_graph_matches_native_sb3_with_nontrivial_statistics(self):
        env = VecNormalize(DummyVecEnv([ContractEnv]), norm_reward=False, clip_obs=10.)
        env.obs_rms.mean = np.linspace(-1., 1., 61)
        env.obs_rms.var = np.linspace(0.01, 5., 61)
        model = PPO('MlpPolicy', env, n_steps=2, batch_size=2, device='cpu', seed=42)
        actor = NormalizedSB3Actor(model.policy, env)
        batch = np.random.default_rng(42).normal(size=(32, 61)).astype(np.float32)
        batch[16:] *= 1000
        expected, _ = model.predict(env.normalize_obs(batch.copy()), deterministic=True)
        with torch.no_grad():
            actual = actor(torch.from_numpy(batch)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertFalse(np.allclose(actual[:16], actor.policy._predict(torch.from_numpy(batch[:16]), deterministic=True).detach().numpy()))
        env.close()


if __name__ == '__main__':
    unittest.main()
