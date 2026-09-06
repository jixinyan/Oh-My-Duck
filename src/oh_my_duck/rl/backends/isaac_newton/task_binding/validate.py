"""Run a headless shared-task physics/MDP gate before enabling a trainable binding."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.backends.mujoco.registration import build_environment
    from oh_my_duck.rl.backends.isaac_newton.task_binding.environment import NewtonTaskEnvironment

    task = project_tasks().get(args.task)
    cfg = build_environment(task.binding("mujoco"))
    cfg.scene.num_envs = 64
    cfg.seed = 42
    env = None
    try:
        env = NewtonTaskEnvironment(cfg, task=task, device="cuda:0")
        from .audit import audit_collisions, audit_randomization

        physics_audit = audit_collisions(env, task.model)
        dr_audit = audit_randomization(env)
        obs, _ = env.reset()
        assert obs["actor"].shape == (64, 61), obs["actor"].shape
        reference = env.scene["robot"]
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch

        native = as_torch(env.native.scene["robot"].data.root_pose_w)
        torch.testing.assert_close(reference.data.root_link_pos_w, native[:, :3], atol=1e-5, rtol=1e-5)
        motor = env.native.scene["robot"].actuators["official_bam"]
        calls = motor.physics_calls
        penalty_names = {
            name
            for name, term in cfg.rewards.items()
            if term is not None
            and (term.weight < 0 or getattr(term.func, "__name__", "").endswith(("_penalty", "_l1")))
        }
        ranges = {}
        for i in range(120):
            obs, reward, terminated, truncated, info = env.step(torch.zeros(64, 14, device="cuda:0"))
            if (
                not all(torch.isfinite(value).all() for value in obs.values())
                or not torch.isfinite(reward).all()
            ):
                raise FloatingPointError(f"Nonfinite task state at step {i}")
            assert motor.physics_calls - calls == (i + 1) * cfg.decimation
            values = env.reward_manager._step_reward
            for index, name in enumerate(env.reward_manager.active_terms):
                low, high = float(values[:, index].min()), float(values[:, index].max())
                assert torch.isfinite(values[:, index]).all(), name
                if name in penalty_names:
                    assert high <= 1e-7, (name, high)
                previous = ranges.get(name, (low, high))
                ranges[name] = [min(previous[0], low), max(previous[1], high)]
        report = {
            "task": args.task,
            "physics": "isaac-newton",
            "num_envs": 64,
            "steps": 120,
            "actor_shape": list(obs["actor"].shape),
            "physics_audit": physics_audit,
            "dr_audit": dr_audit,
            "bam_calls_per_action": cfg.decimation,
            "reward_ranges": ranges,
            "reward_terms": env.reward_manager.active_terms,
            "sensors": list(env.scene.sensors),
            "status": "physics_mdp_smoke_only",
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        import traceback

        (args.output / "failure.json").write_text(
            json.dumps({"error": repr(error), "traceback": traceback.format_exc()}, indent=2) + "\n"
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
