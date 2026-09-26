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
    if spec['framework'] == 'sb3':
        from oh_my_duck.rl.learners.sb3.checkpoint import load_resume_metadata
        metadata = load_resume_metadata(run, recovery['checkpoint'])
    else:
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
        steps = metadata['timesteps_after']
        block = spec['num_envs'] * 24
        if steps % block:
            raise ValueError('Checkpoint is not on a complete rollout boundary')
        completed = steps // block
    else:
        progress = json.loads(subprocess.check_output([
            str(root / '.envs/mujoco/bin/python'),
            '-m', 'oh_my_duck.rl.learners.rsl_rl.progress',
            '--checkpoint', str(checkpoint),
            '--agent-config', str(run / 'params/agent.yaml'),
        ], text=True))
        completed = progress['completed_iterations']
    if completed != recovery['completed_iterations']:
        raise ValueError('Declared progress differs from native checkpoint')
    remaining = spec['iterations'] - completed
    if remaining <= 0:
        raise ValueError('No training budget remains')
    return {**recovery, 'remaining_iterations': remaining,
            'native_progress': progress if spec['framework'] == 'rsl-rl' else None,
            'state_restoration': 'native optimizer, normalizer and curriculum; fresh simulator episodes'}
