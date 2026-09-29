"""Extract the reproduction scope from the pinned official registry on CPU."""
import argparse
import json
from pathlib import Path
import mjlab.tasks
from oh_my_duck.rl.training.tasks import project_tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from oh_my_duck.rl.backends.mujoco.registration import register_tasks
register_tasks()

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def name(value):
    return getattr(value, '__module__', type(value).__module__) + '.' + getattr(value, '__qualname__', type(value).__qualname__)


def catalog():
    tasks = []
    for spec in project_tasks().list('mujoco'):
        # Keep the generated official inventory free of project-only variants.
        if spec.maturity != "inventory":
            continue
        task = spec.id
        cfg, agent = load_env_cfg(task), load_rl_cfg(task)
        model = cfg.scene.entities['robot'].build().compile()
        servos = [model.joint(i).name for i in range(model.njnt)
                  if model.jnt_type[i] != 0 and not model.joint(i).name.startswith('passive_')]
        if len(servos) != 14:
            raise ValueError(f'{task}: expected 14 named servos, got {servos}')
        tasks.append({'id': task, 'backlash': 'Backlash' in task,
            'rough': 'Rough' in task, 'physics_dt': cfg.sim.mujoco.timestep,
            'decimation': cfg.decimation, 'num_steps_per_env': agent.num_steps_per_env,
            'official_max_iterations': agent.max_iterations,
            'model': {'nq': model.nq, 'nv': model.nv, 'bodies': model.nbody,
                      'passive_joints': [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith('passive_')]},
            'observations': {group: list(terms.terms) for group, terms in cfg.observations.items()},
            'commands': {key: name(value) for key, value in cfg.commands.items()},
            'rewards': {key: {'function': name(value.func), 'weight': value.weight}
                        for key, value in cfg.rewards.items() if value is not None},
            'events': list(cfg.events), 'terminations': list(cfg.terminations), 'curriculum': list(cfg.curriculum),
            'sensors': [name(sensor) for sensor in cfg.scene.sensors]})
    return {'schema_version': 1, 'source': json.loads((ROOT / 'configs/upstream.json').read_text())['repositories']['microduck_rl'],
            'tasks': tasks, 'representative_tasks': json.loads((ROOT / 'configs/training.json').read_text())['representative_tasks'],
            'target_combinations': len(json.loads((ROOT / 'configs/training.json').read_text())['representative_tasks']) * 4,
            'note': 'Registry inventory only. Registration does not establish training or behavior validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = catalog()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Recorded {len(result['tasks'])} official tasks / {result['target_combinations']} target combinations")


if __name__ == '__main__':
    main()
