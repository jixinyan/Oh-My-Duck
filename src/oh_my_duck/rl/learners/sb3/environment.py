"""SB3 boundary for the pinned official mjlab task, including delayed terminal obs."""
from copy import copy, deepcopy

import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from stable_baselines3.common.vec_env import VecEnv
from mjlab.managers.recorder_manager import RecorderTerm


class TerminalObservationRecorder(RecorderTerm):
    """Evaluate final actor state without advancing the live delay/history buffers.

    mjlab caches compute(update_history=False), so calling it here would return
    obs_t, not obs_(t+1). A private copy of the observation buffers is essential.
    No task, reward, command, DR or reset implementation is replaced.
    """
    def record_pre_reset(self, env_ids):
        env = self._env
        manager = copy(env.observation_manager)
        for field in ("_group_obs_term_delay_buffer", "_group_obs_term_history_buffer",
                      "_group_obs_class_instances"):
            setattr(manager, field, deepcopy(getattr(manager, field)))
        # Noise/lag sampling for the snapshot must not consume the training RNG.
        devices = [torch.device(env.device)] if str(env.device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            env.sim.forward()
            env.sim.sense()
            groups = getattr(env, "_sb3_observation_groups", ("actor",))
            final = {key: manager.compute_group(key, update_history=True)[env_ids].detach().clone()
                     for key in groups}
        states = final if "critic" in groups else final["actor"]
        env._sb3_terminal = (env_ids.detach().clone(), states)


class MjlabSb3VecEnv(VecEnv):
    def __init__(self, env, *, critic_observations="actor"):
        if critic_observations not in {"actor", "official"}:
            raise ValueError("Unknown critic observation layout")
        self.env = env
        self.critic_observations = critic_observations
        env._sb3_observation_groups = ("actor", "critic") if critic_observations == "official" else ("actor",)
        self.render_mode = None
        self._pending_actions = None
        self.terminal_count = self.timeout_count = 0
        self.episode_returns = np.zeros(env.num_envs)
        self.episode_lengths = np.zeros(env.num_envs, dtype=np.int64)
        obs = env.single_observation_space.spaces["actor"]
        if obs.shape != (61,) or env.single_action_space.shape != (14,):
            raise ValueError("Microduck requires 61 actor observations and 14 servo actions")
        # Gym requires finite bounds. Use the float32 representable domain; no
        # additional practical action clamp or filter is introduced.
        limit = np.finfo(np.float32).max
        space = Box(-np.inf, np.inf, (61,), np.float32)
        if critic_observations == "official":
            critic = env.single_observation_space.spaces["critic"]
            if len(critic.shape) != 1:
                raise ValueError("SB3 expects a flat official critic group")
            space = Dict({"actor": space, "critic": Box(-np.inf, np.inf, critic.shape, np.float32)})
        super().__init__(env.num_envs, space,
                         Box(-limit, limit, (14,), np.float32))

    def reset(self):
        obs, _ = self.env.reset(seed=self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        self.episode_returns.fill(0)
        self.episode_lengths.fill(0)
        return self._observations(obs)

    @staticmethod
    def _numpy(value):
        if isinstance(value, dict):
            return {key: MjlabSb3VecEnv._numpy(item) for key, item in value.items()}
        return value.detach().cpu().numpy().copy()

    def _observations(self, obs):
        if getattr(self, "critic_observations", "actor") == "official":
            return self._numpy({key: obs[key] for key in ("actor", "critic")})
        return self._numpy(obs["actor"])

    def step_async(self, actions):
        self._pending_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        self.env._sb3_terminal = None
        obs, reward, terminated, truncated, extras = self.env.step(self._pending_actions)
        self._pending_actions = None
        obs, reward = self._observations(obs), self._numpy(reward)
        terminated, truncated = self._numpy(terminated), self._numpy(truncated)
        values = obs.values() if isinstance(obs, dict) else (obs,)
        if not all(np.isfinite(value).all() for value in values) or not np.isfinite(reward).all():
            raise FloatingPointError("Non-finite SB3 transition")
        self.episode_returns += reward
        self.episode_lengths += 1
        dones = terminated | truncated
        infos = [{"TimeLimit.truncated": bool(truncated[i] and not terminated[i])}
                 for i in range(self.num_envs)]
        ids = np.flatnonzero(dones)
        if len(ids):
            snapshot = self.env._sb3_terminal
            if snapshot is None:
                raise RuntimeError("Missing pre-reset terminal observation")
            snapshot_ids, states = map(self._numpy, snapshot)
            np.testing.assert_array_equal(snapshot_ids, ids)
            groups = states if isinstance(states, dict) else {"actor": states}
            for key, value in groups.items():
                dim = self.observation_space[key].shape[0] if isinstance(states, dict) else 61
                if value.shape != (len(ids), dim) or not np.isfinite(value).all():
                    raise FloatingPointError(f"Invalid terminal observation: {key}")
            for row, i in enumerate(ids):
                infos[i].update(terminal_observation={k: v[row].copy() for k, v in states.items()} if isinstance(states, dict) else states[row].copy(),
                    episode={"r": float(self.episode_returns[i]), "l": int(self.episode_lengths[i])})
            self.terminal_count += len(ids)
            self.timeout_count += sum(infos[i]["TimeLimit.truncated"] for i in ids)
            self.episode_returns[ids] = 0
            self.episode_lengths[ids] = 0
        self.last_log = extras.get("log", {})
        return obs, reward, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        value = self.render_mode if attr_name == "render_mode" else getattr(self.env, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Change official task configuration before constructing the environment")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("Per-environment method dispatch is not supported by mjlab")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
