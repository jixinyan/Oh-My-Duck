import json
from pathlib import Path
import tempfile
import unittest
from oh_my_duck.rl.experiments.preview import checkpoint_source


class PreviewCheckpointSelection(unittest.TestCase):
    def test_partial_sb3_save_cannot_replace_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); campaign=root/'campaign'; run=campaign/'stand'; run.mkdir(parents=True)
            output=root/'training'; complete=output/'checkpoints/step_000100'; complete.mkdir(parents=True)
            for name in ('model.zip','vecnormalize.pkl','run.json'):
                (complete/name).write_text('{}')
            partial=output/'checkpoints/.step_000200.partial'; partial.mkdir(); (partial/'model.zip').write_text('partial')
            incomplete=output/'checkpoints/step_000300'; incomplete.mkdir(); (incomplete/'model.zip').write_text('incomplete')
            (run/'result.json').write_text(json.dumps({'stages':{'full':{'command':['train','--output',str(output)]}}}))
            spec={'id':'stand','framework':'sb3'}
            self.assertEqual(checkpoint_source(spec,campaign,root),(complete,complete/'model.zip'))

    def test_recovered_run_can_preview_original_checkpoint_before_new_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'source'
            spec={'id':'walking','framework':'rsl-rl','resume':{'run':str(run),'checkpoint':'model_250.pt'}}
            self.assertEqual(checkpoint_source(spec,root/'new',root),(run,run/'model_250.pt'))


def test_training_preview_stops_with_its_worker_even_if_training_raises(tmp_path):
    from unittest.mock import patch, MagicMock
    import pytest
    from oh_my_duck.rl.experiments.preview import training_preview
    child = MagicMock(pid=123)
    child.wait.return_value = 0
    output = tmp_path / 'run'
    output.mkdir()
    with patch('oh_my_duck.rl.experiments.preview.subprocess.Popen', return_value=child) as spawn:
        with pytest.raises(RuntimeError, match='training failure'):
            with training_preview(tmp_path, output, {'id':'walker'}, '3', 1000) as record:
                assert record['gpu'] == '3'
                raise RuntimeError('training failure')
    assert (output / 'preview-training-finished').exists()
    assert record['status'] == 'completed'
    child.wait.assert_called_once()
    command = spawn.call_args.args[0]
    assert command[command.index('--gpu')+1] == '3'
    assert command[command.index('--minimum-updates')+1] == '1000'
    assert not spawn.call_args.kwargs.get('start_new_session',False)


def test_independent_worker_preview_reads_own_manifest(tmp_path):
    from contextlib import ExitStack
    import time
    from oh_my_duck.core.paths import project_root
    from oh_my_duck.rl.experiments.preview import training_preview

    (tmp_path / 'campaign.json').write_text(json.dumps({'status': 'running', 'plan': {'runs': []}}))
    outputs = [tmp_path / name for name in ('standup-seed42', 'standup-seed43')]
    specs = [{'id': output.name, 'framework': 'sb3'} for output in outputs]
    for output in outputs:
        output.mkdir()
    with ExitStack() as stack:
        records = [stack.enter_context(training_preview(
            project_root(), output, spec, '0', 250, poll_interval=0.05,
        )) for output, spec in zip(outputs, specs)]
        for output in outputs:
            report = output / 'previews' / 'previews.json'
            deadline = time.monotonic() + 5
            while not report.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert report.exists(), (output / 'preview.log').read_text()
        time.sleep(0.15)
        assert all(record['status'] == 'running' for record in records)
    for output, spec, record in zip(outputs, specs, records):
        assert record['status'] == 'completed', (output / 'preview.log').read_text()
        manifest = json.loads((output / 'preview-campaign' / 'campaign.json').read_text())
        assert manifest['plan']['runs'] == [spec]
        assert Path(manifest['run_output']) == output.resolve()
    assert json.loads((tmp_path / 'campaign.json').read_text())['plan']['runs'] == []


def test_independent_worker_checkpoint_source_uses_run_output(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    output = tmp_path / 'independent-run'
    output.mkdir()
    training = tmp_path / 'training'
    bundle = training / 'checkpoints' / 'step_000001'
    bundle.mkdir(parents=True)
    environment = VecNormalize(DummyVecEnv([lambda: gym.make('Pendulum-v1')]), norm_reward=False)
    model = PPO('MlpPolicy', environment, n_steps=24, batch_size=24, n_epochs=1, device='cpu')
    model.learn(total_timesteps=24)
    model.save(bundle / 'model.zip')
    environment.save(bundle / 'vecnormalize.pkl')
    (bundle / 'run.json').write_text(json.dumps({
        'timesteps_after': model.num_timesteps, 'num_envs': environment.num_envs,
    }))
    environment.close()
    (output / 'result.json').write_text(json.dumps({
        'stages': {'full': {'command': ['train', '--output', str(training)]}},
    }))
    spec = {'id': 'standup-seed43', 'framework': 'sb3'}
    assert checkpoint_source(spec, tmp_path / 'preview-campaign', tmp_path,
                             minimum_updates=1, run_output=output) == (bundle, bundle / 'model.zip')


def test_campaign_preview_cli_reads_existing_campaign(tmp_path):
    import subprocess
    import sys
    from oh_my_duck.core.paths import project_root

    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    spec = {'id': 'walking', 'framework': 'sb3'}
    (campaign / 'campaign.json').write_text(json.dumps({
        'status': 'completed', 'plan': {'runs': [spec]},
    }))
    output = tmp_path / 'campaign-previews'
    result = subprocess.run([
        sys.executable, str(project_root() / 'omd.py'), 'preview',
        '--campaign', str(campaign), '--output', str(output), '--gpu', '0',
    ], cwd=project_root(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / 'previews.json').read_text())['runs'] == {}
