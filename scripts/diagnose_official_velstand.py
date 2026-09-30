import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from oh_my_duck.rl.evaluation.rehearsal import infer_policy as ip
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
from oh_my_duck.robotics.microduck.protocol import HOME


def run(scene: Path, policy_path: Path, speed: float, stand_ticks: int,
        move_ticks: int, stop_ticks: int, spawn_x: float, spawn_y: float) -> dict:
    bam = ip.load_bam_model(200.0, 7.4, ip.BAM_MAX_CURRENT)
    model, data, controller, _ = ip.load_mujoco_with_bam(str(scene), bam, 0.005, 0.1, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(model, data, walking_onnx_path=str(policy_path),
                                action_scale=1.0, bam_ctrl=controller,
                                use_projected_gravity=True, new_cmd_obs=True)
    mujoco.mj_resetData(model, data)
    root = int(model.jnt_qposadr[model.joint("trunk_base_freejoint").id])
    data.qpos[root:root + 7] = [spawn_x, spawn_y, 0.125, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = HOME
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    report = {"scene": str(scene), "spawn_xy_m": [spawn_x, spawn_y],
              "model": {"timestep_s": model.opt.timestep, "body_count": model.nbody,
                        "geom_count": model.ngeom, "contact_geom_count": int(np.count_nonzero(model.geom_contype)),
                        "actuator_count": model.nu,
                        "foot_friction": {name: model.geom_friction[model.geom(name).id].tolist()
                                          for name in ("left_foot_collision", "right_foot_collision")}},
              "stages": {}}
    for name, count, command in (("stand", stand_ticks, (0.0, 0.0, 0.0)),
                                 ("move", move_ticks, (speed, 0.0, 0.0)),
                                 ("stop", stop_ticks, (0.0, 0.0, 0.0))):
        policy.set_vel_cmd(*command)
        positions, local_twists, world_twists, actions, contacts, observations = [], [], [], [], [], []
        targets, joint_velocities, quaternions, freejoint_velocities = [], [], [], []
        body_twists = []
        for _ in range(count):
            observation = policy.get_observations()
            action = policy.infer()
            if observation.shape != (61,) or action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError("Invalid official policy inference")
            policy.apply_action(action)
            targets.append(controller.q_target.copy())
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("Invalid BAM physical state")
            velocity = np.zeros(6)
            world_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     policy.trunk_base_id, velocity, 1)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     policy.trunk_base_id, world_velocity, 0)
            positions.append(data.qpos[root:root + 3].copy())
            local_twists.append(velocity[[3, 4, 5, 0, 1, 2]].copy())
            world_twists.append(world_velocity[[3, 4, 5, 0, 1, 2]].copy())
            actions.append(action.copy())
            contacts.append(data.ncon)
            observations.append(observation.copy())
            joint_velocities.append(data.qvel[policy.joint_qvel_indices].copy())
            quaternions.append(data.qpos[root + 3:root + 7].copy())
            freejoint_velocities.append(data.qvel[:6].copy())
            body_linear = policy.quat_rotate_inverse(
                data.qpos[root + 3:root + 7].astype(np.float32), data.qvel[:3].astype(np.float32))
            body_twists.append([float(body_linear[0]), float(body_linear[1]),
                                float(data.qvel[5])])
        positions = np.asarray(positions)
        local_twists = np.asarray(local_twists)
        world_twists = np.asarray(world_twists)
        actions = np.asarray(actions)
        observations = np.asarray(observations)
        targets = np.asarray(targets)
        joint_velocities = np.asarray(joint_velocities)
        quaternions = np.asarray(quaternions)
        freejoint_velocities = np.asarray(freejoint_velocities)
        body_twists = np.asarray(body_twists)
        report["stages"][name] = {"ticks": count, "start_position_m": positions[0].tolist(),
                                   "end_position_m": positions[-1].tolist(),
                                   "displacement_xy_m": (positions[-1, :2] - positions[0, :2]).tolist(),
                                   "mean_local_twist": local_twists.mean(axis=0).tolist(),
                                   "mean_world_twist": world_twists.mean(axis=0).tolist(),
                                   "max_abs_local_twist": np.max(np.abs(local_twists), axis=0).tolist(),
                                   "contact_count_min": int(min(contacts)),
                                   "contact_count_max": int(max(contacts)),
                                   "observed_command_first": observations[0, 48:61].tolist(),
                                   "observed_command_last": observations[-1, 48:61].tolist(),
                                   "first_target_rad": targets[0].tolist(),
                                   "target_range_rad": [float(targets.min()), float(targets.max())],
                                   "joint_velocity_max_abs_rad_s": float(np.max(np.abs(joint_velocities))),
                                   "initial_quaternion_wxyz": quaternions[0].tolist(),
                                   "final_quaternion_wxyz": quaternions[-1].tolist(),
                                   "mean_freejoint_qvel": freejoint_velocities.mean(axis=0).tolist(),
                                   "mean_body_twist_from_freejoint": body_twists.mean(axis=0).tolist(),
                                   "action_min": float(actions.min()),
                                   "action_max": float(actions.max())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--policy", choices=("velstand", "alpha_walking"), default="velstand")
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--stand-ticks", type=int, default=50)
    parser.add_argument("--move-ticks", type=int, default=100)
    parser.add_argument("--stop-ticks", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    policy_path = root / ".cache" / "official-policies" / OFFICIAL_REVISION / f"{args.policy}.onnx"
    scenes = root / "src" / "oh_my_duck" / "robotics" / "microduck" / "microduck"
    baseline = "scene.xml" if args.policy == "velstand" else "scene_walk.xml"
    results = {name: run(scenes / name, policy_path, args.speed,
                         args.stand_ticks, args.move_ticks, args.stop_ticks,
                         args.spawn_x, args.spawn_y)
               for name in (baseline, "scene_apartment.xml")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
