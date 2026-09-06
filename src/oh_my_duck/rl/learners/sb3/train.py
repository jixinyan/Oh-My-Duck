"""Train PPO with SB3 using the project-owned Microduck task registry."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import mjlab.tasks  # official entry-point registration, including MDP safety patches
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from oh_my_duck.rl.backends.mujoco.registration import register_tasks
register_tasks()
from mjlab.utils.torch import configure_torch_backends
from oh_my_duck.rl.training.tasks import project_tasks
from oh_my_duck.rl.training.runtime import create_environment
from oh_my_duck.rl.tasks.recipes import build_environment
from oh_my_duck.rl.learners.sb3.environment import MjlabSb3VecEnv, TerminalObservationRecorder

from oh_my_duck.core.paths import project_root
ROOT = project_root()
from oh_my_duck.infrastructure.tracking import start_run


class RewardAudit(BaseCallback):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.ranges = {}
        self.step_ranges = {}
        self.step_min = self.step_max = None
        self.penalties = {name for name, term in adapter.env.cfg.rewards.items()
                          if term is not None and (term.weight < 0 or
                              getattr(term.func, "__name__", "").endswith(("_penalty", "_l1")))}

    def _on_step(self):
        # Audit every active term even when the smoke is shorter than an episode.
        # Keep reductions on GPU and transfer only at the rollout boundary.
        values = self.adapter.env.reward_manager._step_reward.detach()
        low, high = values.amin(dim=0), values.amax(dim=0)
        self.step_min = low if self.step_min is None else torch.minimum(self.step_min, low)
        self.step_max = high if self.step_max is None else torch.maximum(self.step_max, high)
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

    def _on_rollout_end(self):
        low, high = self.step_min.cpu().numpy(), self.step_max.cpu().numpy()
        for i, name in enumerate(self.adapter.env.reward_manager.active_terms):
            if not np.isfinite([low[i], high[i]]).all():
                raise FloatingPointError(f"Non-finite weighted reward: {name}")
            if name in self.penalties and high[i] > 1e-7:
                raise ValueError(f"Positive weighted penalty: {name}={high[i]}")
            self.step_ranges[name] = [float(low[i]), float(high[i])]
            self.logger.record("Step_Reward/min/" + name, float(low[i]))
            self.logger.record("Step_Reward/max/" + name, float(high[i]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[t.id for t in project_tasks().list("mujoco")])
    parser.add_argument("--backend", default="mujoco")
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
    from oh_my_duck.infrastructure.provenance import source_provenance, validate_resume
    provenance = source_provenance()
    task = project_tasks().get(args.task)
    binding = task.binding(args.backend)
    cfg, official_agent = build_environment(binding), binding.rsl_config.build()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    cfg.recorders["sb3_terminal"] = RecorderTermCfg(func=TerminalObservationRecorder)
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = tracking = None
    started = time.monotonic()
    try:
        tracking = start_run(backend=args.backend, framework="sb3", task=args.task, directory=args.output,
            config={"num_envs": args.num_envs, "iterations": args.iterations, "seed": args.seed,
                    "resume_source": str(args.resume) if args.resume else None,
                    "official_agent": asdict(official_agent), "provenance":provenance})
        adapter = MjlabSb3VecEnv(create_environment(task,cfg,backend=args.backend,device=args.device))
        progress_source = "fresh_environment"
        if args.resume:
            previous = json.loads((args.resume / "run.json").read_text())
            validate_resume(previous,task=args.task,backend=args.backend,framework="sb3")
            if previous["backend"] != args.backend or previous["task"] != args.task or previous["upstream"] != json.loads((ROOT / "configs/upstream.json").read_text())["repositories"]:
                raise ValueError("Resume task or upstream pin differs")
            normalized = VecNormalize.load(args.resume / "vecnormalize.pkl", adapter)
            model = PPO.load(args.resume / "model.zip", env=normalized, device=args.device)
            from .checkpoint import restore_progress
            adapter.env.common_step_counter, progress_source = restore_progress(previous, model.num_timesteps)
        else:
            normalized = VecNormalize(adapter, norm_obs=True, norm_reward=False, clip_obs=100.0)
            algorithm = official_agent.algorithm
            policy_cfg = task.policy_configs["sb3"].build(task_id=args.task,agent_cfg=official_agent)
            model = PPO(policy_cfg.policy, normalized, n_steps=official_agent.num_steps_per_env,
                batch_size=args.num_envs * official_agent.num_steps_per_env // algorithm.num_mini_batches,
                n_epochs=algorithm.num_learning_epochs, learning_rate=algorithm.learning_rate,
                gamma=algorithm.gamma, gae_lambda=algorithm.lam, clip_range=algorithm.clip_param,
                ent_coef=algorithm.entropy_coef, vf_coef=algorithm.value_loss_coef,
                max_grad_norm=algorithm.max_grad_norm, target_kl=algorithm.desired_kl,
                policy_kwargs=policy_cfg.kwargs,
                device=args.device, seed=args.seed, verbose=1, tensorboard_log=str(args.output / "tensorboard"))
        model.tensorboard_log = str(args.output / "tensorboard")
        normalized.training = True
        callback = RewardAudit(adapter)
        before = model.num_timesteps
        environment_step_before = adapter.env.common_step_counter
        model.learn(total_timesteps=args.iterations * args.num_envs * official_agent.num_steps_per_env,
                    callback=callback, reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "model.zip")
        normalized.save(args.output / "vecnormalize.pkl")
        # Verify native reload including normalization, not just ZIP existence.
        batch = np.random.default_rng(args.seed).normal(size=(16, 61)).astype(np.float32)
        batch[:, 3:6] = (0, 0, -1)
        # Official configure_torch_backends enables TF32 for training. Compare
        # reload in FP32 so GPU tensor-core rounding is not mistaken for lost state.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        expected, _ = model.predict(normalized.normalize_obs(batch.copy()), deterministic=True)
        loaded_norm = VecNormalize.load(args.output / "vecnormalize.pkl", adapter)
        loaded = PPO.load(args.output / "model.zip", device="cpu")
        actual, _ = loaded.predict(loaded_norm.normalize_obs(batch.copy()), deterministic=True)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        report = {"framework": "sb3", "backend": args.backend, "task": args.task,
            "num_envs": args.num_envs, "iterations": args.iterations,
            "timesteps_before": before, "timesteps_after": model.num_timesteps,
            "resume": str(args.resume) if args.resume else None,
            "env_state_before": {"common_step_counter": environment_step_before},
            "env_state": {"common_step_counter": adapter.env.common_step_counter},
            "env_state_restore": progress_source,
            "episode_length_s_override": args.episode_length_s,
            "terminal_snapshots": adapter.terminal_count, "timeouts": adapter.timeout_count,
            "reward_ranges": callback.ranges, "step_reward_ranges": callback.step_ranges, "reload_max_abs_error": float(np.max(np.abs(actual - expected))),
            "upstream": json.loads((ROOT / "configs/upstream.json").read_text())["repositories"],
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "official_agent": asdict(official_agent),
            "differences": ["SB3 PPO implementation and optimizer; constant learning rate with target-KL stop",
                "SB3 critic uses actor observations; official privileged critic is not consumed",
                "SB3 VecNormalize running statistics and clipping (100); reward normalization disabled",
                "Finite float32 action-space bounds, no additional action filter",
                "Terminal observation sampled before reset from copied delay/history buffers"],
            "policy_status": "trained; behavior_unvalidated; export_requires_separate_gate",
            "wall_time_s": time.monotonic() - started, "provenance":provenance,
            "wandb": {"id": tracking.id, "url": tracking.url, "mode": tracking.settings.mode}}
        report["files"] = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                           for name in ("model.zip", "vecnormalize.pkl")}
        (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        tracking.summary.update({"terminal_snapshots": adapter.terminal_count, "timeouts": adapter.timeout_count,
            "reload_max_abs_error": report["reload_max_abs_error"], "status": "passed"})
        print(json.dumps(report, indent=2))
    except Exception as error:
        if tracking is not None:
            tracking.summary["status"] = "failed"
        (args.output / "failure.json").write_text(json.dumps({"status": "failed",
            "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if tracking is not None:
            tracking.finish(exit_code=1 if (args.output / "failure.json").exists() else 0)


if __name__ == "__main__":
    main()
