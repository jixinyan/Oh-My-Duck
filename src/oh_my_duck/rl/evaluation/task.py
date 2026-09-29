"""Replay a normalized ONNX policy in a registered task, with metrics and native video."""

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def main():
    from oh_my_duck.infrastructure.headless import configure_egl

    configure_egl()
    from oh_my_duck.rl.evaluation.mujoco_video import MujocoVideo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--mujoco-renderer", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("standard", "training-stage"), default="standard")
    parser.add_argument("--curriculum-step", type=int, help="Freeze the official curriculum at this control-step count (diagnostic only)")
    parser.add_argument("--push-scale", type=float, default=1.0, help="Explicit diagnostic push amplitude multiplier; timing remains unchanged")
    parser.add_argument("--record-rewards", action="store_true")
    parser.add_argument("--diagnostic-forward-speed", type=float,
                        help="Override only Walking's forward command (±0.4 m/s); diagnostic, never acceptance")
    args = parser.parse_args()
    import math
    if (args.profile == "training-stage") != (args.curriculum_step is not None):
        parser.error("training-stage requires --curriculum-step; standard forbids it")
    if (args.curriculum_step is not None and args.curriculum_step < 0) or not math.isfinite(args.push_scale) or args.push_scale < 0:
        parser.error("Use nonnegative finite diagnostic parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from oh_my_duck.rl.artifacts.inference import cpu_session
    import imageio.v2 as imageio
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.training.runtime import create_environment
    from oh_my_duck.rl.tasks.recipes import build_environment
    from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES
    from oh_my_duck.rl.backends.mujoco.registration import register_tasks

    register_tasks()
    task = project_tasks().get(args.task)
    if task.evaluation is None:
        raise ValueError(f"Task {task.id} has no evaluation protocol")
    from .diagnostics import diagnostic_forward_protocol
    protocol = diagnostic_forward_protocol(task.evaluation.build(), args.diagnostic_forward_speed)
    cfg = build_environment(task.binding(args.backend), play=args.profile == "standard")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    from .diagnostics import prepare_recipe, apply_stage, summarize_rewards, summarize_motion
    stages = prepare_recipe(cfg, profile=args.profile, curriculum_step=args.curriculum_step)
    cfg.episode_length_s = 1 + max(len(s.commands) for s in protocol.scenarios) * 0.02
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 0.85, 135.0, -20.0
    assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
    session = cpu_session(str(args.policy), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 61] and session.get_outputs()[0].shape == [1, 14]
    assert tuple(session.get_modelmeta().custom_metadata_map["joint_names"].split(",")) == JOINT_NAMES
    report = {
        "task": task.id,
        "backend": args.backend,
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "auto_reset": False,
        "mujoco_renderer": args.mujoco_renderer,
        "render_resolution": [args.width, args.height],
        "renderer": "isolated-native-mujoco" if args.backend == "mujoco" else "native-newton",
        "seed": args.seed,
        "scenarios": {},
        "evaluation_profile": args.profile,
        "curriculum_step": args.curriculum_step,
        "push_scale": args.push_scale,
        "diagnostic_forward_speed": args.diagnostic_forward_speed,
        "acceptance_eligible": (args.profile == "standard" and args.push_scale == 1.0
                                and args.diagnostic_forward_speed is None),
        "reward_trace_units": "weighted rate before policy-dt integration" if args.record_rewards else None,
    }
    env = writer = video = None
    try:
        env = create_environment(
            task,
            cfg,
            backend=args.backend,
            device="cuda:0",
            render_mode="rgb_array" if args.video and args.backend != "mujoco" else None,
        )
        # Standard defaults retain the original evaluation behavior. Diagnostic
        # stage configuration is applied before forced-pose resets.
        if stages or args.push_scale != 1.0:
            weights = apply_stage(env, stages, args.curriculum_step, args.push_scale)
        else:
            weights = {name: env.reward_manager.get_term_cfg(name).weight
                       for name in env.reward_manager.active_terms}
        if args.record_rewards:
            report["reward_weights"] = weights
            report["reward_term_order"] = list(env.reward_manager.active_terms)
        if args.video and args.backend == "mujoco":
            video = MujocoVideo(
                env.sim.mj_model, width=args.width, height=args.height, renderer=args.mujoco_renderer
            )
        for index, scenario in enumerate(protocol.scenarios):
            if scenario.reset_probabilities is not None:
                event = env.event_manager.get_term_cfg("set_ground_state")
                event.params.update(scenario.reset_probabilities)
            obs, _ = env.reset(seed=args.seed + index)
            trace = defaultdict(list)
            if args.video:
                writer = imageio.get_writer(
                    args.output / (scenario.name + ".mp4"), fps=25, codec="libx264", macro_block_size=1
                )
            completed = True
            for step, command in enumerate(scenario.commands):
                state = obs["actor"].clone()
                state[:, 48:] = 0.0
                state[:, 48:51] = torch.tensor(command, device=env.device)
                for name in env.command_manager.active_terms:
                    value = env.command_manager.get_command(name)
                    value.zero_()
                    if name == "twist":
                        value[:] = torch.tensor(command, device=env.device)
                assert torch.isfinite(state).all()
                action = session.run(None, {session.get_inputs()[0].name: state.cpu().numpy()})[0]
                assert action.shape == (1, 14) and np.isfinite(action).all()
                robot = env.scene["robot"].data
                height = float((robot.root_link_pos_w - env.scene.env_origins)[0, 2])
                tilt = float(torch.acos((-robot.projected_gravity_b[0, 2]).clamp(-1, 1)))
                twist = (
                    torch.cat((robot.root_link_lin_vel_b[0, :2], robot.root_link_ang_vel_b[0, 2:3]))
                    .cpu()
                    .numpy()
                )
                for key, value in {
                    "obs": state.cpu().numpy()[0],
                    "action": action[0],
                    "height": height,
                    "tilt": tilt,
                    "twist": twist,
                    "command": command,
                }.items():
                    trace[key].append(value)
                if args.record_rewards:
                    for key, value in {
                        "joint_position": robot.joint_pos[0],
                        "joint_velocity": robot.joint_vel[0],
                        "base_position": robot.root_link_pos_w[0] - env.scene.env_origins[0],
                        "base_quaternion": robot.root_link_quat_w[0],
                    }.items():
                        trace[key].append(value.detach().cpu().numpy().copy())
                if writer is not None and step % 2 == 0:
                    if video is not None:
                        frame = video.render(
                            env.sim.data.qpos[0].cpu().numpy(),
                            env.sim.data.qvel[0].cpu().numpy(),
                            robot.root_link_pos_w[0].cpu().numpy(),
                        )
                    else:
                        frame = env.render()
                    if (
                        frame.shape != (args.height, args.width, 3)
                        or not np.isfinite(frame).all()
                        or np.ptp(frame.astype(float)) < 1
                    ):
                        imageio.imwrite(args.output / (scenario.name + "_invalid.png"), frame)
                        raise ValueError(
                            f"Invalid rendered frame: shape={frame.shape}, range={np.ptp(frame.astype(float))}"
                        )
                    writer.append_data(frame)
                    if step == 0:
                        imageio.imwrite(args.output / (scenario.name + ".png"), frame)
                with torch.no_grad():
                    obs, reward, terminated, truncated, info = env.step(
                        torch.from_numpy(action).to(env.device)
                    )
                if args.record_rewards:
                    trace["weighted_reward_rate"].append(env.reward_manager._step_reward[0].detach().cpu().numpy().copy())
                    trace["transition_reward"].append(float(reward[0]))
                if not torch.isfinite(reward).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                    raise FloatingPointError("Nonfinite replay state")
                if terminated.any() or truncated.any():
                    completed = False
                    break
            if writer is not None:
                writer.close()
                writer = None
            np.savez_compressed(
                args.output / (scenario.name + ".npz"),
                **{key: np.asarray(value) for key, value in trace.items()},
            )
            report["scenarios"][scenario.name] = {
                "ticks": len(trace["height"]),
                "completed": completed,
                **protocol.score(trace, completed),
            }
            if args.record_rewards:
                report["scenarios"][scenario.name]["reward_terms"] = summarize_rewards(trace, report["reward_term_order"])
                if protocol.kind == "walking":
                    report["scenarios"][scenario.name]["motion_diagnostics"] = summarize_motion(trace)
        passed = all(s["success"] for s in report["scenarios"].values())
        report["status"] = "passed" if passed else "behavior_failed"
        if not report["acceptance_eligible"]:
            report["status"] = "diagnostic_" + report["status"]
    except Exception as error:
        import traceback

        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
            if video is not None:
                video.close()
        except Exception as cleanup_error:
            report["cleanup_error"] = repr(cleanup_error)
            if report.get("status") != "error":
                report["status"] = "error"
                raise
        finally:
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"passed", "diagnostic_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
