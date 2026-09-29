"""Campaign configuration reaches each native SB3 training stage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.experiments import worker
from oh_my_duck.rl.experiments.campaign import load_plan


class CampaignLearningRate(unittest.TestCase):
    def plan(self):
        plan = json.loads((project_root() / 'configs/experiments/representative-sb3-lr-1e-4.json').read_text())
        plan['runs'] = [plan['runs'][0]]
        return plan

    def test_invalid_or_wrong_framework_options_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            for rate in (0, -0.1, True, '0.0001', None, float('inf'), float('nan')):
                with self.subTest(rate=rate):
                    plan = self.plan()
                    plan['runs'][0]['learning_rate'] = rate
                    path.write_text(json.dumps(plan))
                    with self.assertRaisesRegex(ValueError, 'learning_rate'):
                        load_plan(path)
            plan = self.plan()
            plan['runs'][0]['framework'] = 'rsl-rl'
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'only to native SB3'):
                load_plan(path)

    def test_worker_applies_rate_to_smoke_resume_capacity_and_full(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = self.plan()
                if not explicit:
                    del plan['runs'][0]['learning_rate']
                if explicit:
                    plan['runs'][0].update(learning_rate_mode='adaptive', critic_observations='official', initial_episode_phase='randomized', action_rate_delay_iterations=4500, low_speed_tracking_boost=1.0)
                config = root / 'plan.json'
                config.write_text(json.dumps(plan))
                output = root / 'run'
                observed = []

                def execute(command, **kwargs):
                    # Replace only process execution: exercise real plan parsing,
                    # stage construction, resume selection and report writing.
                    if 'train' in command:
                        observed.append(command)
                        destination = Path(command[command.index('--output') + 1])
                        destination.mkdir()
                        (destination / 'run.json').write_text(json.dumps({'wall_time_s': 1.0}))
                    return 0

                argv = ['worker', '--config', str(config), '--run-id', plan['runs'][0]['id'], '--output', str(output)]
                with patch('sys.argv', argv), patch.object(worker.subprocess, 'call', side_effect=execute):
                    self.assertEqual(worker.main(), 0)
                self.assertEqual(len(observed), 4)
                self.assertEqual([Path(c[c.index('--output') + 1]).name for c in observed],
                                 ['smoke', 'resume-check', 'capacity', 'full'])
                self.assertIn('--resume', observed[1])
                for command in observed:
                    self.assertEqual('--learning-rate' in command, explicit)
                    self.assertEqual('--action-rate-delay-iterations' in command, explicit)
                    self.assertEqual('--low-speed-tracking-boost' in command, explicit)
                    if explicit:
                        self.assertEqual(command[command.index('--action-rate-delay-iterations')+1], '4500')
                        self.assertEqual(float(command[command.index('--low-speed-tracking-boost')+1]), 1.0)
                        self.assertEqual(float(command[command.index('--learning-rate') + 1]), 0.0001)
                        for key,value in [('learning-rate-mode','adaptive'),('critic-observations','official'),('initial-episode-phase','randomized')]:
                            self.assertEqual(command[command.index('--'+key)+1],value)
                saved = json.loads((output / 'result.json').read_text())
                self.assertEqual(saved['spec'], plan['runs'][0])
