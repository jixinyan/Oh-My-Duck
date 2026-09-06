"""Official BAM three-second noisy HOME settle, with measured height and tilt."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.common.protocol import HOME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from omd_microduck.rehearsal import infer_policy as ip
    rng = np.random.default_rng(42)
    records = []
    for seed in range(args.samples):
        bam = ip.load_bam_model(ip.BAM_KP_FW, float(rng.uniform(*ip.BAM_VIN_RANGE)), ip.BAM_MAX_CURRENT)
        model, data, controller, _ = ip.load_mujoco_with_bam(str(ip.MICRODUCK_XML), bam, .005,
            float(rng.uniform(*ip.BAM_VIN_DROP_GAIN_RANGE)), ip.BAM_VIN_MIN)
        joint_ids = model.actuator_trnid[:, 0]
        qids = model.jnt_qposadr[joint_ids]
        base = model.joint('trunk_base_freejoint').qposadr[0]
        data.qpos[base:base+3] = [0, 0, .125]
        angle = rng.uniform(-.05, .05, 3)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, angle, 'xyz')
        data.qpos[base+3:base+7] = quat
        data.qpos[qids] = np.asarray(HOME) + rng.uniform(-.015, .015, 14)
        controller.reset(data.qpos)
        controller.q_target[:] = HOME
        mujoco.mj_forward(model, data)
        body = model.body('trunk_base').id
        max_tilt = 0.
        finite = True
        for _ in range(600):
            controller.update()
            mujoco.mj_step(model, data)
            finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            tilt = float(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
        height = float(data.qpos[base+2])
        records.append({'seed': seed, 'height_m': height, 'tilt_rad': tilt, 'max_tilt_rad': max_tilt,
            'finite': finite, 'passed': finite and height > .065 and max_tilt < np.pi/3})
    result = {'status': 'passed' if all(r['passed'] for r in records) else 'failed',
        'duration_s': 3., 'physics_steps': 600, 'samples': records,
        'physics': 'official CPU MuJoCo/BAM M6', 'note': 'HOME equilibrium only; no learned behavior claim'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status']=='passed' else 2


if __name__ == '__main__':
    sys.exit(main())
