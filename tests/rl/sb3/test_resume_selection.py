import hashlib
import json
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from oh_my_duck.rl.experiments.worker import sb3_training_arguments
from oh_my_duck.rl.learners.sb3.checkpoint import load_resume_metadata, restore_progress


def vector_environment():
    return DummyVecEnv([lambda: gym.make('Pendulum-v1')])


def save_bundle(directory, model, normalizer):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / 'model.zip')
    normalizer.save(directory / 'vecnormalize.pkl')
    metadata = {
        'num_envs': model.n_envs, 'timesteps_before': 0,
        'timesteps_after': model.num_timesteps,
        'normalizer_count': float(normalizer.obs_rms.count),
        'files': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in ('model.zip', 'vecnormalize.pkl')},
    }
    (directory / 'run.json').write_text(json.dumps(metadata))


@pytest.fixture
def native_bundles(tmp_path):
    torch.set_num_threads(1)
    normalizer = VecNormalize(vector_environment(), norm_reward=False)
    model = PPO('MlpPolicy', normalizer, n_steps=24, batch_size=24, n_epochs=1,
                policy_kwargs={'net_arch': [16]}, device='cpu', seed=42)
    run = tmp_path / 'run'
    periodic = run / 'checkpoints' / 'step_000000000048'
    try:
        model.learn(48)
        save_bundle(periodic, model, normalizer)
        model.learn(24, reset_num_timesteps=False)
        save_bundle(run, model, normalizer)
        yield run, periodic, model, normalizer
    finally:
        normalizer.close()


@pytest.mark.parametrize('source', ['final', 'periodic'])
def test_native_resume_uses_selected_bundle_and_completes_budget(native_bundles, tmp_path, source):
    run, periodic, final_model, final_normalizer = native_bundles
    selected = run if source == 'final' else periodic
    metadata = load_resume_metadata(selected)
    completed = metadata['timesteps_after'] // 24
    target_iterations = 5
    command = sb3_training_arguments(
        {'seed': 42}, tmp_path / 'continued', 1, target_iterations - completed, 2,
        selected, 'model.zip')
    actual = Path(command[command.index('--resume') + 1])
    assert actual == selected.resolve()
    loaded_metadata = load_resume_metadata(actual)
    normalizer = VecNormalize.load(actual / 'vecnormalize.pkl', vector_environment())
    try:
        model = PPO.load(actual / 'model.zip', env=normalizer, device='cpu')
        assert model.num_timesteps == (72 if source == 'final' else 48)
        assert model._n_updates == completed
        assert restore_progress(loaded_metadata, model.num_timesteps)[0] == model.num_timesteps
        assert normalizer.obs_rms.count == loaded_metadata['normalizer_count']
        if source == 'final':
            np.testing.assert_array_equal(normalizer.obs_rms.mean, final_normalizer.obs_rms.mean)
            np.testing.assert_array_equal(normalizer.obs_rms.var, final_normalizer.obs_rms.var)
            for actual_parameter, expected_parameter in zip(model.policy.parameters(), final_model.policy.parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
            expected_optimizer = final_model.policy.optimizer.state_dict()
            actual_optimizer = model.policy.optimizer.state_dict()
            assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
            for key, state in expected_optimizer['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual_optimizer['state'][key][name], value, rtol=0, atol=0)
        remaining = int(command[command.index('--iterations') + 1])
        model.learn(remaining * 24, reset_num_timesteps=False)
        assert model.num_timesteps == target_iterations * 24
        assert model._n_updates == target_iterations
    finally:
        normalizer.close()


@pytest.mark.parametrize('filename', ['model.zip', 'vecnormalize.pkl'])
def test_resume_rejects_changed_native_bundle(native_bundles, filename):
    run, periodic, _, _ = native_bundles
    shutil.copyfile(periodic / filename, run / filename)
    with pytest.raises(ValueError, match='hash changed'):
        load_resume_metadata(run)


def test_resume_rejects_missing_selected_file_without_selecting_periodic(native_bundles):
    run, periodic, _, _ = native_bundles
    (run / 'model.zip').unlink()
    assert (periodic / 'model.zip').is_file()
    with pytest.raises(FileNotFoundError):
        load_resume_metadata(run)


def test_resume_rejects_another_checkpoint_filename(native_bundles, tmp_path):
    run, _, _, _ = native_bundles
    with pytest.raises(ValueError, match='must be model.zip'):
        sb3_training_arguments({'seed': 42}, tmp_path / 'continued', 1, 2, 2, run, 'other.zip')
    with pytest.raises(ValueError, match='must be model.zip'):
        load_resume_metadata(run, 'other.zip')


def test_resume_rejects_inconsistent_progress(native_bundles):
    run, _, _, _ = native_bundles
    metadata = load_resume_metadata(run)
    metadata['timesteps_after'] += 24
    model = PPO.load(run / 'model.zip', device='cpu')
    with pytest.raises(ValueError, match='does not match'):
        restore_progress(metadata, model.num_timesteps)
