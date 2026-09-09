"""Privileged critic isolation, native PPO timeouts, save/reload and deployment."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from oh_my_duck.rl.models.sb3_asymmetric import AsymmetricActorCriticPolicy
from oh_my_duck.rl.artifacts.sb3_export import NormalizedSB3Actor
from oh_my_duck.rl.learners.sb3.environment import TerminalObservationRecorder


class GroupEnv(gym.Env):
    observation_space = Dict({"actor": Box(-np.inf, np.inf, (61,), np.float32),
                              "critic": Box(-np.inf, np.inf, (74,), np.float32)})
    action_space = Box(-1000, 1000, (14,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation(), {}

    def observation(self):
        return {"actor": np.full(61, self.t * .1, dtype=np.float32),
                "critic": np.full(74, self.t * .7 + 2, dtype=np.float32)}

    def step(self, action):
        self.t += 1
        return self.observation(), float(1 - np.square(action).mean()), False, self.t == 3, {}


def make_model():
    env = VecNormalize(DummyVecEnv([GroupEnv, GroupEnv]), norm_reward=False)
    return PPO(AsymmetricActorCriticPolicy, env, n_steps=4, batch_size=4, n_epochs=2,
               policy_kwargs={"net_arch": {"pi": [16], "vf": [24]}, "activation_fn": torch.nn.ELU},
               seed=42, device="cpu"), env


def test_critic_cannot_change_actor_or_receive_actor_gradients():
    model, env = make_model()
    obs = {"actor": torch.randn(8, 61, requires_grad=True), "critic": torch.randn(8, 74, requires_grad=True)}
    actions, values, _ = model.policy(obs, deterministic=True)
    other = {**obs, "critic": obs["critic"] + 20}
    actions_other, values_other, _ = model.policy(other, deterministic=True)
    torch.testing.assert_close(actions, actions_other, atol=0, rtol=0)
    assert not torch.allclose(values, values_other)
    actions.sum().backward()
    assert torch.count_nonzero(obs["critic"].grad) == 0
    assert torch.count_nonzero(obs["actor"].grad) > 0
    torch.testing.assert_close(actions, model.policy.actor_mean(obs["actor"]))
    env.close()


def test_native_dict_ppo_reloads_and_exports_only_actor_statistics():
    model, env = make_model()
    model.learn(32)
    assert isinstance(model.rollout_buffer, DictRolloutBuffer)
    assert model.rollout_buffer.observations["critic"].shape[-1] == 74
    # Deliberately distinct statistics catch accidental actor/critic mixing.
    env.obs_rms["actor"].mean = np.linspace(-1, 1, 61)
    env.obs_rms["actor"].var = np.linspace(.01, 5, 61)
    obs = {"actor": np.random.default_rng(1).normal(size=(16, 61)).astype(np.float32),
           "critic": np.full((16, 74), 99, np.float32)}
    obs["actor"][8:] *= 1000
    expected, _ = model.predict(env.normalize_obs(obs), deterministic=True)
    with tempfile.TemporaryDirectory() as d:
        model.save(Path(d)/"model.zip")
        env.save(Path(d)/"vecnormalize.pkl")
        loaded = PPO.load(Path(d)/"model.zip", device="cpu")
        norm = VecNormalize.load(Path(d)/"vecnormalize.pkl", DummyVecEnv([GroupEnv, GroupEnv]))
        actual, _ = loaded.predict(norm.normalize_obs(obs), deterministic=True)
        np.testing.assert_allclose(actual, expected)
        graph = NormalizedSB3Actor(loaded.policy, norm)
        with torch.no_grad():
            output = graph(torch.from_numpy(obs["actor"])).numpy()
        np.testing.assert_allclose(output, expected, atol=1e-6, rtol=1e-5)
        norm.close()
    env.close()


def test_native_timeout_bootstrap_consumes_normalized_terminal_critic():
    model, env = make_model()
    predictions = []
    native = model.policy.predict_values
    def record(obs):
        value = native(obs)
        if obs["critic"].shape[0] == 1:
            predictions.append((obs["critic"].detach().clone(), float(value.item())))
        return value
    model.policy.predict_values = record
    class Capture(BaseCallback):
        def _on_step(self):
            if self.locals["dones"].any():
                self.raw_rewards = self.locals["rewards"].copy()
                self.terminals = [x["terminal_observation"] for x in self.locals["infos"]]
            return True
        def _on_rollout_end(self):
            self.corrected = self.model.rollout_buffer.rewards[2].copy()
    callback = Capture()
    model.learn(8, callback=callback)
    assert len(predictions) == 2
    for i, (critic, value) in enumerate(predictions):
        np.testing.assert_allclose(critic.numpy()[0], callback.terminals[i]["critic"])
        np.testing.assert_allclose(callback.corrected[i], callback.raw_rewards[i] + model.gamma * value)
    env.close()


def actor_state(env):
    return env.actor


def critic_state(env):
    return env.critic


def test_terminal_recorder_copies_both_groups_without_mutating_live_history():
    env = SimpleNamespace(num_envs=2, device="cpu", actor=torch.ones(2,61), critic=torch.ones(2,74),
                          _sb3_observation_groups=("actor", "critic"),
                          sim=SimpleNamespace(forward=lambda: None, sense=lambda: None))
    cfg = {k: ObservationGroupCfg(terms={"state": ObservationTermCfg(func=f, delay_min_lag=1, delay_max_lag=1)})
           for k, f in [("actor",actor_state),("critic",critic_state)]}
    manager = env.observation_manager = ObservationManager(cfg, env)
    manager.compute(update_history=True)
    env.actor.fill_(2); env.critic.fill_(12); manager.compute(update_history=True)
    env.actor.fill_(3); env.critic.fill_(13)
    cached = {k:v.clone() for k,v in manager.compute(update_history=False).items()}
    recorder = TerminalObservationRecorder(RecorderTermCfg(func=TerminalObservationRecorder),env)
    recorder.record_pre_reset(torch.tensor([1]))
    _, final = env._sb3_terminal
    assert final["actor"].shape == (1,61) and final["critic"].shape == (1,74)
    assert final["actor"][0,0] == 2 and final["critic"][0,0] == 12
    for k,v in manager.compute(update_history=False).items():
        torch.testing.assert_close(v,cached[k])


def test_official_initial_episode_phase_spreads_timeouts_reproducibly():
    from oh_my_duck.rl.learners.sb3.environment import MjlabSb3VecEnv
    n=64
    env=SimpleNamespace(num_envs=n,device="cpu",max_episode_length=300,
        single_observation_space=GroupEnv.observation_space,
        single_action_space=GroupEnv.action_space,episode_length_buf=torch.zeros(n,dtype=torch.long))
    def reset(seed):
        torch.manual_seed(seed)
        env.episode_length_buf.zero_()
        return {"actor":torch.zeros(n,61),"critic":torch.zeros(n,74)},{}
    env.reset=reset
    wrapper=MjlabSb3VecEnv(env,critic_observations="official",initial_episode_phase="randomized")
    wrapper.seed(42);wrapper.reset();first=env.episode_length_buf.clone()
    assert len(first.unique())>32 and first.max()<300
    wrapper.seed(42);wrapper.reset();torch.testing.assert_close(first,env.episode_length_buf)
    wrapper.initial_episode_phase="synchronized";wrapper.seed(42);wrapper.reset()
    assert env.episode_length_buf.count_nonzero()==0
