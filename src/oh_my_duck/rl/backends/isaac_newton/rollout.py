"""Finite Newton diagnostic rollout with measured state and optional offscreen video."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from oh_my_duck.rl.backends.isaac_newton.paths import asset_dir
from oh_my_duck.rl.backends.isaac_newton.contracts import JOINT_NAMES


def main(mode="eval"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--actuator", choices=["pd", "bam"], default="pd")
    parser.add_argument("--schedule", type=Path, help="Shared command sequence for policy evaluation")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    if min(args.video_width, args.video_height) < 2 or args.video_width % 2 or args.video_height % 2:
        parser.error("Video dimensions must be positive even integers of at least 2 pixels")
    schedule = None
    if args.schedule:
        from oh_my_duck.rl.backends.isaac_newton.contracts import protocol
        schedule = protocol.compile_schedule(json.loads(args.schedule.read_text()))
        args.steps = len(schedule)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "Raw-asset PD diagnostic; not BAM locomotion or sim2sim",
              "planned_ticks": args.steps, "recorded_ticks": 0, "mode": mode,
              "render_resolution": [args.video_width, args.video_height]}
    if args.actuator == "bam":
        report["scope"] = "Official BAM policy replay on converted asset; complete task/contact parity pending"
    env = writer = None
    trace = []
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from isaaclab_tasks.utils import launch_simulation
        from oh_my_duck.rl.backends.isaac_newton.config import DiagnosticEnvCfg
        from oh_my_duck.rl.backends.isaac_newton.environment import DiagnosticEnv
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch
        from oh_my_duck.rl.backends.isaac_newton.physics import check_raw_asset, inspect_solver
        if not torch.cuda.is_available():
            raise RuntimeError("No allocated CUDA device; run through a server job")
        cfg = DiagnosticEnvCfg()
        cfg.scene.num_envs = args.num_envs
        if args.actuator == "bam":
            from oh_my_duck.rl.backends.isaac_newton.bam_actuator import OfficialBamActuatorCfg
            from oh_my_duck.rl.backends.isaac_newton.walk_asset import configure_walk_scene
            configure_walk_scene(cfg.scene)
            cfg.scene.robot.actuators = {"official_bam": OfficialBamActuatorCfg(joint_names_expr=list(JOINT_NAMES))}
        # Keep the actual failed state available; assess it explicitly below.
        cfg.terminations.fallen = None
        cfg.episode_length_s = (args.steps + 2) * 0.02
        if args.video or mode == "probe":
            from isaaclab.sensors import CameraCfg
            from isaaclab.sim import PinholeCameraCfg
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            cfg.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", width=args.video_width, height=args.video_height,
                data_types=["rgb"], update_period=0.0, update_latest_camera_pose=True, spawn=PinholeCameraCfg(clipping_range=(0.01, 10.0)),
                renderer_cfg=NewtonWarpRendererCfg())
        with launch_simulation(cfg, {"headless": True}):
            env = DiagnosticEnv(cfg)
            obs, _ = env.reset()
            report.update(inspect_solver(env))
            armature_override = None
            if args.actuator == "bam":
                armature_override = float(env.scene["robot"].actuators["official_bam"].armature[0, 0])
            report["asset_checks"] = check_raw_asset(env, json.loads((asset_dir() / "reference.json").read_text()), armature_override)
            if args.actuator == "bam":
                from oh_my_duck.rl.backends.isaac_newton.collision_checks import check_walk_collisions
                report["collision_checks"] = check_walk_collisions(env)
            report["packages"] = {n: importlib.metadata.version(n) for n in ("isaaclab", "newton", "mujoco-warp", "warp-lang", "torch")}
            session = None
            if args.policy:
                from oh_my_duck.rl.artifacts.inference import cpu_session
                session = cpu_session(str(args.policy), providers=["CPUExecutionProvider"])
                if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
                    raise ValueError("Expected a 61-input/14-output policy")
                metadata = session.get_modelmeta().custom_metadata_map
                scales = np.asarray([float(x) for x in metadata.get("action_scale", "nan").split(",")])
                if tuple(metadata.get("joint_names", "").split(",")) != JOINT_NAMES or scales.size not in (1, 14) or not np.all(scales == 1.0):
                    raise ValueError("Policy metadata does not match canonical joint order and scale")
                report["policy_sha256"] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(args.output / "rollout.mp4", fps=25)
            for step in range(args.steps):
                state = obs["policy"].clone()
                command = schedule[step][2] if schedule else (0., 0., 0.)
                state[:, 48:51] = torch.tensor(command, device=env.device)
                if tuple(state.shape) != (args.num_envs, 61) or not torch.isfinite(state).all():
                    raise ValueError("Malformed or non-finite observation")
                if session:
                    actions = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                        for row in state.detach().cpu().numpy()])
                    action = torch.from_numpy(actions).to(env.device)
                else:
                    action = torch.zeros((args.num_envs, 14), device=env.device)
                if not torch.isfinite(action).all():
                    raise ValueError("Non-finite policy action")
                robot = env.scene["robot"]
                trace.append({"obs": state.detach().cpu().numpy().copy(), "action": action.cpu().numpy().copy(),
                    "root_pos": as_torch(robot.data.root_pos_w).cpu().numpy().copy(),
                    "twist": torch.cat((as_torch(robot.data.root_lin_vel_b)[:, :2], as_torch(robot.data.root_ang_vel_b)[:, 2:3]), dim=-1).cpu().numpy().copy(),
                    "command": np.asarray(command),
                    "joint_pos": as_torch(robot.data.joint_pos).cpu().numpy().copy()})
                if (args.video or mode == "probe") and step % 2 == 0:
                    camera = env.scene["camera"]
                    lookat = as_torch(robot.data.root_pos_w).clone()
                    eye = lookat + torch.tensor([0.6, 0.6, 0.35], device=env.device)
                    camera.set_world_poses_from_view(eye, lookat)
                    env.sim.render()
                    camera.update(0.0, force_recompute=True)
                    np.testing.assert_allclose(as_torch(camera.data.pos_w).cpu().numpy(),
                        eye.cpu().numpy(), atol=1e-5, rtol=0)
                    frame = as_torch(camera.data.output["rgb"])[0, ..., :3].cpu().numpy()
                    if not np.isfinite(frame).all() or np.ptp(frame.astype(float)) < 1:
                        raise ValueError("Offscreen frame is empty or uniform")
                    if step == 0:
                        import imageio.v3 as iio
                        iio.imwrite(args.output / "frame.png", frame)
                    if writer:
                        writer.append_data(frame)
                with torch.inference_mode():
                    obs, reward, terminated, truncated, info = env.step(action)
                report["recorded_ticks"] += 1
                from oh_my_duck.rl.backends.isaac_newton.mdp import fallen
                if fallen(env).any():
                    report["fallen_height_m"] = as_torch(robot.data.root_pos_w)[:, 2].cpu().tolist()
                    report["fallen_gravity"] = as_torch(robot.data.projected_gravity_b).cpu().tolist()
                    raise RuntimeError(f"Policy replay fell at tick {step + 1}")
                if terminated.any() or truncated.any():
                    raise RuntimeError(f"Diagnostic terminated at tick {step + 1}; automatic reset is not a pass")
            measured = np.stack([item["twist"] for item in trace])
            commanded = np.stack([item["command"] for item in trace])[:, None, :]
            report["twist_rmse"] = np.sqrt(np.mean((measured-commanded)**2, axis=(0, 1))).tolist()
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        except Exception as error:
            report["status"] = "failed"
            report["teardown_failure"] = str(error)
        if trace:
            import numpy as np
            np.savez_compressed(args.output / "trajectory.npz", **{key: np.stack([row[key] for row in trace]) for key in trace[0]})
        report["wall_time_s"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1
