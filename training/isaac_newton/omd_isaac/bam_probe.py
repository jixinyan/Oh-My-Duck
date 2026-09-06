"""Three-second noisy HOME settle using official BAM on Newton; no RL claim."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "scope": "BAM actuator and noisy HOME settle; task migration not validated"}
    env = None
    try:
        from isaaclab_tasks.utils import launch_simulation
        from .config import DiagnosticEnvCfg
        from .mdp import as_torch
        from .environment import DiagnosticEnv
        from .bam_actuator import OfficialBamActuatorCfg
        cfg = DiagnosticEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.scene.robot.actuators = {"official_bam": OfficialBamActuatorCfg(joint_names_expr=list(JOINT_NAMES))}
        cfg.events.joints.params["position_range"] = (-0.015, 0.015)
        cfg.events.root.params["pose_range"] = {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05)}
        cfg.episode_length_s = 4.0
        with launch_simulation(cfg, {"headless": True}):
            env = DiagnosticEnv(cfg)
            obs, _ = env.reset()
            motor = env.scene["robot"].actuators["official_bam"]
            initial_calls = motor.physics_calls
            trace = []
            started = time.monotonic()
            for tick in range(150):
                obs, rewards, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 14), device=env.device))
                state = obs["policy"]
                gravity = state[:, 3:6]
                tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
                height = as_torch(env.scene["robot"].data.root_pos_w)[:, 2]
                trace.append(torch.stack((height, tilt), dim=-1).detach().cpu().numpy())
                if not torch.isfinite(state).all() or not torch.isfinite(motor.applied_effort).all():
                    raise FloatingPointError("Non-finite BAM state")
                if torch.any(terminated | truncated):
                    raise RuntimeError(f"Noisy HOME settle terminated at tick {tick}")
            official = motor.official
            assert motor.physics_calls - initial_calls == 600, "BAM must run at physics substeps, not policy frequency"
            result.update(status="passed", ticks=150, physics_calls=600, num_envs=args.num_envs,
                final_height_m=height.detach().cpu().tolist(), final_tilt_rad=tilt.detach().cpu().tolist(),
                wall_time_s=time.monotonic() - started, solver=type(motor.solver).__name__,
                actuator_type=type(official).__name__, delay_physics_steps=[official.cfg.delay_min_lag, official.cfg.delay_max_lag],
                voltage_range=list(official.cfg.vin_range), force_limit=motor.force_limit)
            np.savez_compressed(args.output / "trajectory.npz", height_tilt=np.asarray(trace))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
