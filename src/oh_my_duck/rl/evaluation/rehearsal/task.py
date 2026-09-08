"""Official CPU MuJoCo/BAM deployment rehearsal for registered task batteries."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math


def main():
    from oh_my_duck.infrastructure.headless import configure_egl

    configure_egl()
    from oh_my_duck.rl.evaluation.mujoco_video import MujocoVideo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--mujoco-renderer", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio
    from . import infer_policy as ip
    from .state import sample_ground_pose
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.tasks.recipes import build_environment
    from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES, HOME

    task = project_tasks().get(args.task)
    if task.evaluation is None or task.model not in ("walk", "groundcontact"):
        raise ValueError("No CPU deployment scene or evaluation profile registered for this task")
    profile = task.evaluation.build()
    cfg = build_environment(task.binding("mujoco"), play=True)
    # The official deployment rehearsal intentionally uses the full ground-contact
    # scene for both walking and recovery, as does the maintained infer_policy.
    scene = Path(ip.MICRODUCK_XML)
    bam = ip.load_bam_model(200.0, 7.4, ip.BAM_MAX_CURRENT)
    model, data, controller, _ = ip.load_mujoco_with_bam(str(scene), bam, 0.005, 0.1, ip.BAM_VIN_MIN)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    renderer = (
        MujocoVideo(model, width=args.width, height=args.height, renderer=args.mujoco_renderer)
        if args.video
        else None
    )
    writer = None
    report = {
        "task": task.id,
        "backend": "cpu-mujoco-bam-rehearsal",
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "seed": args.seed,
        "auto_reset": False,
        "mujoco_renderer": args.mujoco_renderer,
        "scenarios": {},
        "bam_controller": type(controller).__module__ + "." + type(controller).__name__,
        "friction_constraint_index": "dof",
        "physics_dt": 0.005,
        "policy_dt": 0.02,
        "differences": [
            "Official deployment full groundcontact scene",
            "Fixed 7.4 V and 0.1 sag gain; no training DR or observation corruption",
            "No additional action filtering or sampled training delay",
        ],
    }
    try:
        for index, scenario in enumerate(profile.scenarios):
            np.random.seed(args.seed + index)
            torch.manual_seed(args.seed + index)
            mujoco.mj_resetData(model, data)
            policy = ip.PolicyInference(
                model,
                data,
                walking_onnx_path=str(args.policy),
                action_scale=1.0,
                bam_ctrl=controller,
                use_projected_gravity=True,
                new_cmd_obs=True,
            )
            assert (
                tuple(policy.ort_session.get_modelmeta().custom_metadata_map["joint_names"].split(","))
                == JOINT_NAMES
            )
            np.testing.assert_allclose(policy.default_pose, HOME, atol=1e-7)
            q = int(model.jnt_qposadr[model.joint("trunk_base_freejoint").id])
            data.qpos[q : q + 7] = [0, 0, 0.125, 1, 0, 0, 0]
            data.qpos[policy.joint_qpos_indices] = HOME
            if scenario.reset_probabilities:
                params = cfg.events["set_ground_state"].params.copy()
                params.update(scenario.reset_probabilities)
                sample_ground_pose(model, data, params)
            controller.reset(data.qpos)
            policy.set_position_targets(policy.default_pose)
            mujoco.mj_forward(model, data)
            trace = defaultdict(list)
            completed = True
            if renderer is not None:
                writer = imageio.get_writer(
                    args.output / (scenario.name + ".mp4"), fps=25, codec="libx264", macro_block_size=1
                )
            for step, command in enumerate(scenario.commands):
                policy.set_vel_cmd(*command)
                obs = policy.get_observations().copy()
                action = policy.infer()
                if (
                    obs.shape != (61,)
                    or action.shape != (14,)
                    or not np.isfinite(obs).all()
                    or not np.isfinite(action).all()
                ):
                    raise ValueError("Malformed/nonfinite inference state")
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, policy.trunk_base_id, velocity, 1
                )
                tilt = math.acos(float(np.clip(-obs[5], -1, 1)))
                height = float(data.qpos[q + 2])
                for key, value in {
                    "obs": obs,
                    "action": action,
                    "height": height,
                    "tilt": tilt,
                    "command": command,
                    "twist": velocity[[3, 4, 2]],
                    "qpos": data.qpos.copy(),
                    "qvel": data.qvel.copy(),
                }.items():
                    trace[key].append(value)
                if writer is not None and step % 2 == 0:
                    frame = renderer.render(data.qpos, data.qvel, data.xpos[policy.trunk_base_id])
                    if (
                        frame.shape != (args.height, args.width, 3)
                        or not np.isfinite(frame).all()
                        or np.ptp(frame.astype(float)) < 1
                    ):
                        imageio.imwrite(args.output / (scenario.name + "_invalid.png"), frame)
                        raise ValueError("CPU rehearsal produced an empty rendered frame")
                    writer.append_data(frame)
                    if step == 0:
                        imageio.imwrite(args.output / (scenario.name + ".png"), frame)
                if profile.kind == "walking" and (
                    height < profile.minimum_height or tilt > profile.maximum_tilt
                ):
                    completed = False
                    break
                policy.apply_action(action)
                for _ in range(4):
                    controller.update()
                    mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise FloatingPointError("Nonfinite CPU physics")
                    for warning in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"):
                        if data.warning[getattr(mujoco.mjtWarning, warning)].number:
                            raise FloatingPointError(warning)
            if writer is not None:
                writer.close()
                writer = None
            np.savez_compressed(
                args.output / (scenario.name + ".npz"), **{k: np.asarray(v) for k, v in trace.items()}
            )
            report["scenarios"][scenario.name] = {
                "ticks": len(trace["height"]),
                "completed": completed,
                **profile.score(trace, completed),
            }
        report["status"] = (
            "passed" if all(s["success"] for s in report["scenarios"].values()) else "behavior_failed"
        )
    except Exception as error:
        import traceback

        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
            if renderer is not None:
                renderer.close()
        except Exception as cleanup_error:
            report["cleanup_error"] = repr(cleanup_error)
            if report.get("status") != "error":
                report["status"] = "error"
                raise
        finally:
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
