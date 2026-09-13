"""Verify native checkpoint identity and prior gates before continuing a campaign."""
import hashlib
import json
from pathlib import Path
import subprocess


def validate_checkpoint(spec, root):
    recovery = spec['resume']
    run = Path(recovery['run'])
    checkpoint = run / recovery['checkpoint']
    if Path(recovery['checkpoint']).name != recovery['checkpoint']:
        raise ValueError('Checkpoint must be a filename within the run')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != recovery['checkpoint_sha256']:
        raise ValueError('Recovery checkpoint hash changed')
    metadata = json.loads((run / 'run.json').read_text())
    for key in ('task', 'backend', 'framework'):
        if metadata.get(key) != spec[key]:
            raise ValueError(f'Recovery identity differs: {key}')
    # Recovery only bypasses gates proven by this exact preceding run.
    source = json.loads((Path(recovery['source_campaign']) / spec['id'] / 'result.json').read_text())
    for key in ('task', 'backend', 'framework', 'num_envs'):
        if source['spec'][key] != spec[key]:
            raise ValueError(f'Recovery source differs: {key}')
    if source['spec'].get('action_rate_delay_iterations', 0) != spec.get('action_rate_delay_iterations', 0):
        raise ValueError('Recovery intervention differs; use a fresh paired run')
    from .campaign import SB3_OPTIONS
    if spec['framework'] == 'sb3':
        for key in ('learning_rate', *SB3_OPTIONS):
            if source['spec'].get(key) != spec.get(key):
                raise ValueError(f'Recovery {key} differs; run fresh gates for the new configuration')
    for stage in ('smoke', 'smoke-export', 'smoke-rehearsal', 'resume-check', 'capacity', 'capacity-export'):
        if source['stages'][stage]['status'] != 'completed':
            raise ValueError(f'Recovery requires completed {stage}')
    if spec['framework'] == 'sb3':
        for name in ('model.zip', 'vecnormalize.pkl'):
            if hashlib.sha256((run / name).read_bytes()).hexdigest() != metadata['files'][name]:
                raise ValueError(f'Recovery bundle hash changed: {name}')
        steps = metadata['timesteps_after']
        block = spec['num_envs'] * 24
        if steps % block:
            raise ValueError('Checkpoint is not on a complete rollout boundary')
        completed = steps // block
    else:
        # Read our own native checkpoint in the locked torch environment, keeping
        # the CLI itself free of simulator/learner imports.
        completed = int(subprocess.check_output([
            str(root / '.envs/mujoco/bin/python'), '-c',
            'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["iter"])',
            str(checkpoint)], text=True).strip())
    if completed != recovery['completed_iterations']:
        raise ValueError('Declared progress differs from native checkpoint')
    remaining = spec['iterations'] - completed
    if remaining <= 0:
        raise ValueError('No training budget remains')
    return {**recovery, 'remaining_iterations': remaining,
            'state_restoration': 'native optimizer, normalizer and curriculum; fresh simulator episodes'}
