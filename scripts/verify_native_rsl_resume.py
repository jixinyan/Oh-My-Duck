import argparse
import json
from pathlib import Path

import torch
import yaml

from oh_my_duck.rl.learners.rsl_rl.progress import checkpoint_progress


def load_checkpoint(path, steps_per_iteration):
    data = torch.load(path, map_location='cpu', weights_only=False)
    progress = checkpoint_progress(
        data['iter'], data['infos']['env_state']['common_step_counter'], steps_per_iteration,
    )
    return data, progress


def optimizer_steps(data):
    return max(int(state['step']) for state in data['optimizer_state_dict']['state'].values())


def verify_run(previous_path, run, steps_per_iteration):
    previous, before = load_checkpoint(previous_path, steps_per_iteration)
    metadata = json.loads((run / 'run.json').read_text())
    if Path(metadata['resume']).resolve() != previous_path.resolve():
        raise ValueError(f'Resume checkpoint differs for {run}')
    if metadata['resume_progress'] != before:
        raise ValueError(f'Resume progress differs for {run}')
    if metadata['iteration_before'] != before['completed_iterations']:
        raise ValueError(f'Resumed iteration differs for {run}')
    if metadata['completed_iterations_after'] != before['completed_iterations'] + 1:
        raise ValueError(f'Completed update count differs for {run}')
    if metadata['num_steps_per_env'] != steps_per_iteration:
        raise ValueError(f'Rollout length differs for {run}')
    checkpoint_path = run / f"model_{before['completed_iterations']}.pt"
    current, after = load_checkpoint(checkpoint_path, steps_per_iteration)
    if after['checkpoint_iteration_label'] != before['completed_iterations']:
        raise ValueError(f'Checkpoint label differs for {run}')
    if after['completed_iterations'] != before['completed_iterations'] + 1:
        raise ValueError(f'Checkpoint update count differs for {run}')
    if after['common_step_counter'] != before['common_step_counter'] + steps_per_iteration:
        raise ValueError(f'Environment counter differs for {run}')
    if optimizer_steps(current) <= optimizer_steps(previous):
        raise ValueError(f'Optimizer progress did not advance for {run}')
    for key in ('actor_state_dict', 'critic_state_dict'):
        if float(current[key]['obs_normalizer.count']) <= float(previous[key]['obs_normalizer.count']):
            raise ValueError(f'Observation normalizer did not advance for {run}: {key}')
    return checkpoint_path, {'run': str(run), 'before': before, 'after': after,
                             'optimizer_step_before': optimizer_steps(previous),
                             'optimizer_step_after': optimizer_steps(current)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--first-run', type=Path, required=True)
    parser.add_argument('--second-run', type=Path, required=True)
    args = parser.parse_args()
    agent_path = args.source_checkpoint.parent / 'params/agent.yaml'
    steps_per_iteration = yaml.full_load(agent_path.read_text())['num_steps_per_env']
    first_checkpoint, first = verify_run(args.source_checkpoint, args.first_run, steps_per_iteration)
    _, second = verify_run(first_checkpoint, args.second_run, steps_per_iteration)
    print(json.dumps({'first': first, 'second': second}, indent=2))


if __name__ == '__main__':
    main()
