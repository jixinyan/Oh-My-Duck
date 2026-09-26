import argparse
import json
from pathlib import Path

import torch


def checkpoint_progress(iteration, common_step_counter, num_steps_per_env):
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError('Native checkpoint iteration must be a nonnegative integer')
    if isinstance(common_step_counter, bool) or not isinstance(common_step_counter, int) or common_step_counter <= 0:
        raise ValueError('Native environment step counter must be a positive integer')
    if isinstance(num_steps_per_env, bool) or not isinstance(num_steps_per_env, int) or num_steps_per_env <= 0:
        raise ValueError('Native rollout length must be a positive integer')
    completed, remainder = divmod(common_step_counter, num_steps_per_env)
    if remainder:
        raise ValueError('Native checkpoint was not saved on a complete rollout boundary')
    if iteration >= completed:
        raise ValueError('Native checkpoint label exceeds its completed rollout count')
    return {
        'checkpoint_iteration_label': iteration,
        'completed_iterations': completed,
        'next_iteration_label': completed,
        'common_step_counter': common_step_counter,
        'historical_label_repeats': completed - iteration - 1,
    }


def load_checkpoint_progress(path, num_steps_per_env):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    counter = checkpoint['infos']['env_state']['common_step_counter']
    return checkpoint_progress(checkpoint['iter'], counter, num_steps_per_env)


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    args = parser.parse_args()
    agent = yaml.full_load(args.agent_config.read_text())
    print(json.dumps(load_checkpoint_progress(args.checkpoint, agent['num_steps_per_env'])))


if __name__ == '__main__':
    main()
