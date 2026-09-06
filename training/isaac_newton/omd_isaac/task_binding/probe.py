"""Capture actual Newton model/state evidence needed by the shared task binding."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    from isaaclab_tasks.utils import launch_simulation
    from omd_isaac.config import DiagnosticEnvCfg
    from omd_isaac.environment import DiagnosticEnv
    from omd_isaac.bam_actuator import OfficialBamActuatorCfg
    from omd_isaac.walk_asset import configure_walk_scene
    from omd_isaac.contracts import JOINT_NAMES
    from omd_isaac.mdp import as_torch
    from omd_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
    reference = MICRODUCK_WALK_ROBOT_CFG.build().compile()
    cfg = DiagnosticEnvCfg()
    cfg.scene.num_envs = 64
    configure_walk_scene(cfg.scene)
    cfg.scene.robot.actuators = {'official_bam': OfficialBamActuatorCfg(joint_names_expr=list(JOINT_NAMES))}
    cfg.terminations.fallen = None
    with launch_simulation(cfg, {'headless': True}):
        env = DiagnosticEnv(cfg)
        try:
            env.reset()
            robot = env.scene['robot']
            act = robot.actuators['official_bam']
            if act.official is None:
                act._initialize_bam()
            solver = act.solver
            model, data = solver.mj_model, solver.mjw_data
            report = {'physics': type(solver).__name__, 'num_envs': 64,
                'native_joints': list(robot.joint_names), 'native_bodies': list(robot.body_names),
                'reference_joints': [reference.joint(i).name for i in range(reference.njnt)],
                'reference_bodies': [reference.body(i).name for i in range(reference.nbody)],
                'reference_geoms': [{'name': reference.geom(i).name, 'body': reference.body(int(reference.geom_bodyid[i])).name,
                    'group': int(reference.geom_group[i]), 'active': bool(reference.geom_contype[i] or reference.geom_conaffinity[i])} for i in range(reference.ngeom)],
                'solver_joints': [model.joint(i).name for i in range(model.njnt)],
                'solver_bodies': [model.body(i).name for i in range(model.nbody)],
                'solver_geoms': [model.geom(i).name for i in range(model.ngeom)],
                'solver_sites': [model.site(i).name for i in range(model.nsite)],
                'solver_sensors': [model.sensor(i).name for i in range(model.nsensor)],
                'nu': model.nu, 'nq': model.nq, 'nv': model.nv,
                'env_origins': as_torch(env.scene.env_origins)[:3].cpu().tolist(),
                'native_root_pose': as_torch(robot.data.root_pose_w)[:3].cpu().tolist(),
                'solver_root_qpos': wp.to_torch(data.qpos)[:3, :7].cpu().tolist(),
                'expanded_fields': {name: list(getattr(solver.mjw_model,name).shape) for name in
                    ('body_mass','body_inertia','body_ipos','body_iquat','geom_friction','pair_friction','dof_armature','dof_frictionloss')}}
            (args.output/'mapping.json').write_text(json.dumps(report,indent=2)+'\n')
            print('Newton model/state mapping captured:', args.output)
        finally:
            env.close()


if __name__ == '__main__':
    main()
