"""Headless CPU MuJoCo/BAM replay using the pinned official inference implementation.

No viewer, keyboard loop, wall-clock command timing, or additional action filter.
Rendering is optional and uses EGL; telemetry describes pre-action state at each tick.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from oh_my_duck.core.paths import project_root
ROOT = project_root()
from oh_my_duck.robotics.microduck.protocol import HOME, JOINT_NAMES, compile_schedule


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/flat_walk.json")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    schedule = compile_schedule(config)
    lock = json.loads((ROOT / "configs/upstream.json").read_text())
    policy_path = (args.policy or ROOT / "artifacts/policies/official" / lock["policy"]["revision"] / "alpha_walking.onnx").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from oh_my_duck.rl.evaluation.rehearsal import infer_policy as ip
    np.random.seed(config["seed"])
    bam = config["bam"]
    scene = Path(ip.MICRODUCK_XML)
    bam_model = ip.load_bam_model(bam["kp_fw"], bam["vin"], ip.BAM_MAX_CURRENT)
    model, data, controller, _ = ip.load_mujoco_with_bam(str(scene), bam_model, config["physics_dt"],
                                                       bam["vin_drop_gain"], ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(model, data, walking_onnx_path=str(policy_path),
                                action_scale=config["action_scale"], bam_ctrl=controller,
                                use_projected_gravity=True, new_cmd_obs=True)
    session = policy.ort_session
    assert len(session.get_inputs()) == len(session.get_outputs()) == 1
    assert session.get_inputs()[0].shape == [1, 61], session.get_inputs()[0].shape
    assert session.get_outputs()[0].shape == [1, 14], session.get_outputs()[0].shape
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in model.actuator_trnid[:, 0]]
    assert tuple(names) == JOINT_NAMES, names
    np.testing.assert_allclose(policy.default_pose, HOME, atol=1e-7)
    fj = model.joint("trunk_base_freejoint").id
    qa = int(model.jnt_qposadr[fj])
    data.qpos[qa:qa + 7] = [0, 0, config["initial_trunk_height_m"], 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    control_dt = config["physics_dt"] * config["decimation"]
    video_cfg = config["video"]
    renderer = writer = None
    if args.video:
        stride = 1.0 / (video_cfg["fps"] * control_dt)
        if not math.isclose(stride, round(stride), abs_tol=1e-8) or stride < 1:
            raise ValueError("Video fps must divide policy frequency")
        renderer = mujoco.Renderer(model, height=video_cfg["height"], width=video_cfg["width"])
        writer = imageio.get_writer(args.output / "rollout.mp4", fps=video_cfg["fps"], codec="libx264", macro_block_size=1)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 0.85, 135, -20
    feet = [model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
    trace = defaultdict(list)
    errors_by_segment = defaultdict(list)
    contact_speeds = []
    status, failure = "completed", None
    current_segment = None
    started = time.perf_counter()
    try:
        for step, (segment_id, segment_name, command) in enumerate(schedule):
            if segment_id != current_segment:
                policy.set_vel_cmd(*command)
                current_segment = segment_id
            # Save the exact input BEFORE infer() updates last_action.
            obs = policy.get_observations().copy()
            if obs.shape != (61,) or not np.isfinite(obs).all():
                raise ValueError("Non-finite or malformed policy observation")
            gravity = obs[3:6]
            tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, policy.trunk_base_id, velocity, 1)
            actual_twist = np.array([velocity[3], velocity[4], velocity[2]])
            contact = []
            for geom_id in feet:
                on_ground = any((c.geom1 == geom_id and model.geom_bodyid[c.geom2] == 0)
                                or (c.geom2 == geom_id and model.geom_bodyid[c.geom1] == 0)
                                for c in data.contact)
                contact.append(on_ground)
                if on_ground:
                    foot_velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, foot_velocity, 0)
                    contact_speeds.append(float(np.linalg.norm(foot_velocity[3:5])))
            if data.qpos[qa + 2] < config["fall"]["min_trunk_height_m"] or tilt > config["fall"]["max_tilt_rad"]:
                status, failure = "fell", {"step": step, "time_s": float(data.time), "tilt_rad": tilt,
                                          "height_m": float(data.qpos[qa + 2])}
                break
            action = policy.infer()
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError("Non-finite or malformed policy action")
            policy.apply_action(action)
            for key, value in {"time_s": data.time, "segment": segment_id, "obs": obs,
                               "action": action, "targets": controller.q_target.copy(),
                               "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                               "twist": actual_twist, "command": command, "foot_contact": contact}.items():
                trace[key].append(value)
            errors_by_segment[segment_name].append(actual_twist - command)
            if renderer is not None and step % round(stride) == 0:
                camera.lookat[:] = data.xpos[policy.trunk_base_id]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            for _ in range(config["decimation"]):
                controller.update()
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Non-finite physics state")
                for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"):
                    if data.warning[getattr(mujoco.mjtWarning, name)].number:
                        raise ValueError(f"MuJoCo reported {name}; refusing a silently reset rollout")
    except Exception as exc:
        status, failure = "error", {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        np.savez_compressed(args.output / "trajectory.npz", **{k: np.asarray(v) for k, v in trace.items()})
        result = {"schema_version": 1, "backend": "mujoco", "engine": "cpu-mujoco-bam-m6",
                  "status": status, "failure": failure, "planned_ticks": len(schedule),
                  "recorded_ticks": len(trace["obs"]), "simulation_time_s": float(data.time),
                  "wall_time_s": time.perf_counter() - started,
                  "rmse_by_segment": {k: np.sqrt(np.mean(np.square(v), axis=0)).tolist() for k, v in errors_by_segment.items()},
                  "rmse_axes": ["vx_m_s", "vy_m_s", "yaw_rad_s"],
                  "mean_contact_foot_horizontal_speed_m_s": float(np.mean(contact_speeds)) if contact_speeds else None,
                  "policy": str(policy_path), "policy_sha256": digest(policy_path),
                  "config_sha256": digest(args.config), "upstream": lock["repositories"],
                  "scene": str(scene.relative_to(ROOT)), "onnx_metadata": session.get_modelmeta().custom_metadata_map,
                  "video": "rollout.mp4" if args.video else None,
                  "note": "Single walking policy, no recovery, no additional action filtering or delay. CPU rehearsal collision model."}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
