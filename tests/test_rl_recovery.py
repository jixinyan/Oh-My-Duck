import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from oh_my_duck.rl.experiments.recovery import validate_checkpoint


class RecoveryIntegrity(unittest.TestCase):
    def test_bundle_identity_progress_and_prior_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'checkpoint'
            run.mkdir()
            for name in ('model.zip', 'vecnormalize.pkl'):
                (run / name).write_bytes(name.encode())
            hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                      for name in ('model.zip', 'vecnormalize.pkl')}
            spec = {'id': 'stand', 'task': 'task', 'backend': 'mujoco',
                    'framework': 'sb3', 'num_envs': 64, 'iterations': 100}
            (run / 'run.json').write_text(json.dumps({**spec, 'files': hashes, 'timesteps_after': 64*24*20}))
            previous = root / 'campaign' / 'stand'
            previous.mkdir(parents=True)
            source = {'spec': dict(spec), 'stages': {stage: {'status': 'completed'} for stage in
                      ('smoke', 'smoke-export', 'smoke-rehearsal', 'resume-check', 'capacity', 'capacity-export')}}
            (previous / 'result.json').write_text(json.dumps(source))
            spec['resume'] = {'run': str(run), 'checkpoint': 'model.zip',
                              'checkpoint_sha256': hashes['model.zip'], 'completed_iterations': 20,
                              'source_campaign': str(previous.parent)}
            self.assertEqual(validate_checkpoint(spec, root)['remaining_iterations'], 80)
            spec['learning_rate'] = 0.0001
            with self.assertRaisesRegex(ValueError, 'learning_rate differs'):
                validate_checkpoint(spec, root)
            source['spec']['learning_rate'] = 0.0001
            (previous / 'result.json').write_text(json.dumps(source))
            self.assertEqual(validate_checkpoint(spec, root)['remaining_iterations'], 80)
            spec['resume']['completed_iterations'] = 19
            with self.assertRaisesRegex(ValueError, 'progress'):
                validate_checkpoint(spec, root)
            spec['resume']['completed_iterations'] = 20
            source['stages']['capacity']['status'] = 'failed'
            (previous / 'result.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'capacity'):
                validate_checkpoint(spec, root)
            source['stages']['capacity']['status'] = 'completed'
            (previous / 'result.json').write_text(json.dumps(source))
            (run / 'vecnormalize.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_checkpoint(spec, root)
