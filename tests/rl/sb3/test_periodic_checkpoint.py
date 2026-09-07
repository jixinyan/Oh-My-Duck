import hashlib
import json
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from oh_my_duck.rl.learners.sb3.snapshots import PeriodicCheckpoint
from oh_my_duck.rl.learners.sb3.checkpoint import restore_progress


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(-100., 100., (61,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (14,), dtype=np.float32)
    common_step_counter = 0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(61, dtype=np.float32), {}
    def step(self, action):
        self.common_step_counter += 1
        return np.full(61, self.common_step_counter/100, dtype=np.float32), float(action[0]), False, False, {}


def test_periodic_bundle_is_resumable_with_normalizer_and_progress(tmp_path):
    env = CounterEnv()
    vector = VecNormalize(DummyVecEnv([lambda:env]), norm_reward=False)
    model = PPO('MlpPolicy', vector, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs={'net_arch':[16]}, device='cpu', seed=42)
    callback = PeriodicCheckpoint(tmp_path, 1, SimpleNamespace(env=env),
                                  {'timesteps_before':0, 'num_envs':1})
    model.learn(24, callback=callback)
    bundles = sorted(tmp_path.glob('step_*'))
    assert len(bundles) == 2  # No untrained checkpoint at step zero.
    last = bundles[-1]
    report = json.loads((last/'run.json').read_text())
    assert report['timesteps_after'] == 16
    assert restore_progress(report, 16) == (16, 'saved_environment_state')
    for name, sha in report['files'].items():
        assert hashlib.sha256((last/name).read_bytes()).hexdigest() == sha
    restored_norm = VecNormalize.load(last/'vecnormalize.pkl', DummyVecEnv([CounterEnv]))
    restored = PPO.load(last/'model.zip', env=restored_norm, device='cpu')
    assert restored.num_timesteps == 16
    assert restored._n_updates == 2
    assert restored_norm.obs_rms.count > 16
    restored.learn(8, reset_num_timesteps=False)
    assert restored.num_timesteps == 24
    restored_norm.close()
    vector.close()
