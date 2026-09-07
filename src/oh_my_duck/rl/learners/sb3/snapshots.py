"""Periodic native model/normalizer bundles, published only after a PPO update."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class PeriodicCheckpoint(BaseCallback):
    def __init__(self, directory, interval, adapter, metadata):
        super().__init__()
        if interval < 1:
            raise ValueError('Checkpoint interval must be positive')
        self.directory = Path(directory)
        self.interval = interval
        self.adapter = adapter
        self.metadata = deepcopy(metadata)
        self.rollouts = 0

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # SB3 invokes this after the previous rollout's train() returns. Saving
        # at rollout_end would persist samples/curricula ahead of the PPO update.
        if self.rollouts and self.rollouts % self.interval == 0:
            self.save_bundle()
        self.rollouts += 1

    def save_bundle(self):
        target = self.directory / f'step_{self.model.num_timesteps:012d}'
        temporary = target.with_name('.' + target.name + '.partial')
        temporary.mkdir(parents=True, exist_ok=False)
        self.model.save(temporary / 'model.zip')
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is None:
            raise ValueError('Native VecNormalize is required for a resumable bundle')
        normalizer.save(temporary / 'vecnormalize.pkl')
        report = {**self.metadata, 'timesteps_after': self.model.num_timesteps,
                  'env_state': {'common_step_counter': self.adapter.env.common_step_counter},
                  'checkpoint_kind': 'periodic_after_ppo_update', 'behavior': 'unvalidated'}
        report['files'] = {name: hashlib.sha256((temporary / name).read_bytes()).hexdigest()
                           for name in ('model.zip', 'vecnormalize.pkl')}
        (temporary / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        temporary.rename(target)
        self.logger.record('checkpoint/timesteps', self.model.num_timesteps)
