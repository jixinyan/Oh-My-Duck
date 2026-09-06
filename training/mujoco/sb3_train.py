"""Train PPO with SB3 using the unchanged official Microduck task registry."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import mjlab.tasks  # official entry-point registration, including MDP safety patches
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from sb3_env import MjlabSb3VecEnv, TerminalObservationRecorder

ROOT = Path(__file__).resolve().parents[2]


class RewardAudit(BaseCallback):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.ranges = {}
        self.penalties = {name for name, term in adapter.env.cfg.rewards.items()
                          if term is not None and (term.weight < 0 or
                              getattr(term.func, "__name__", "").endswith(("_penalty", "_l1")))}

    def _on_step(self):
        for name, value in self.adapter.last_log.items():
            if not name.startswith("Episode_Reward/"):
                continue
            value = float(value)
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite reward term: {name}")
            if name.split("/", 1)[1] in self.penalties and value > 1e-7:
                raise ValueError(f"Penalty has positive weighted reward: {name}={value}")
            low, high = self.ranges.get(name, (value, value))
            self.ranges[name] = [min(low, value), max(high, value)]
            self.logger.record(name, value)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[t for t in list_tasks() if "MicroDuck" in t])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Previous run directory, with native model and VecNormalize")
    parser.add_argument("--episode-length-s", type=float, help="Explicit timeout test override, recorded in provenance")
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")
    configure_torch_backends()
    cfg, official_agent = load_env_cfg(args.task), load_rl_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    cfg.recorders["sb3_terminal"] = RecorderTermCfg(func=TerminalObservationRecorder)
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = None
    try:
        adapter = MjlabSb3VecEnv(ManagerBasedRlEnv(cfg, device=args.device))
        if args.resume:
            previous = json.loads((args.resume / "run.json").read_text())
            if previous["task"] != args.task or previous["upstream"] != json.loads((ROOT / "configs/upstream.json").read_text())["repositories"]:
                raise ValueError("Resume task or upstream pin differs")
            normalized = VecNormalize.load(args.resume / "vecnormalize.pkl", adapter)
            model = PPO.load(args.resume / "model.zip", env=normalized, device=args.device)
        else:
            normalized = VecNormalize(adapter, norm_obs=True, norm_reward=False, clip_obs=100.0)
            algorithm = official_agent.algorithm
            model = PPO("MlpPolicy", normalized, n_steps=official_agent.num_steps_per_env,
                batch_size=args.num_envs * official_agent.num_steps_per_env // algorithm.num_mini_batches,
                n_epochs=algorithm.num_learning_epochs, learning_rate=algorithm.learning_rate,
                gamma=algorithm.gamma, gae_lambda=algorithm.lam, clip_range=algorithm.clip_param,
                ent_coef=algorithm.entropy_coef, vf_coef=algorithm.value_loss_coef,
                max_grad_norm=algorithm.max_grad_norm, target_kl=algorithm.desired_kl,
                policy_kwargs={"activation_fn": torch.nn.ELU,
                    "net_arch": dict(pi=list(official_agent.actor.hidden_dims), vf=list(official_agent.critic.hidden_dims))},
                device=args.device, seed=args.seed, verbose=1)
        normalized.training = True
        callback = RewardAudit(adapter)
        before = model.num_timesteps
        model.learn(total_timesteps=args.iterations * args.num_envs * official_agent.num_steps_per_env,
                    callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "model.zip")
        normalized.save(args.output / "vecnormalize.pkl")
        # Verify native reload including normalization, not just ZIP existence.
        batch = np.random.default_rng(args.seed).normal(size=(16, 61)).astype(np.float32)
        batch[:, 3:6] = (0, 0, -1)
        expected, _ = model.predict(normalized.normalize_obs(batch.copy()), deterministic=True)
        loaded_norm = VecNormalize.load(args.output / "vecnormalize.pkl", adapter)
        loaded = PPO.load(args.output / "model.zip", device="cpu")
        actual, _ = loaded.predict(loaded_norm.normalize_obs(batch.copy()), deterministic=True)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        report = {"framework": "sb3", "backend": "mujoco", "task": args.task,
            "num_envs": args.num_envs, "iterations": args.iterations,
            "timesteps_before": before, "timesteps_after": model.num_timesteps,
            "resume": str(args.resume) if args.resume else None,
            "episode_length_s_override": args.episode_length_s,
            "terminal_snapshots": adapter.terminal_count, "timeouts": adapter.timeout_count,
            "reward_ranges": callback.ranges, "reload_max_abs_error": float(np.max(np.abs(actual - expected))),
            "upstream": json.loads((ROOT / "configs/upstream.json").read_text())["repositories"],
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "official_agent": asdict(official_agent),
            "differences": ["SB3 PPO implementation and optimizer; constant learning rate with target-KL stop",
                "SB3 critic uses actor observations; official privileged critic is not consumed",
                "SB3 VecNormalize running statistics and clipping (100); reward normalization disabled",
                "Finite float32 action-space bounds, no additional action filter",
                "Terminal observation sampled before reset from copied delay/history buffers"],
            "policy_status": "training_smoke_only; official-compatible ONNX export pending"}
        report["files"] = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                           for name in ("model.zip", "vecnormalize.pkl")}
        (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
