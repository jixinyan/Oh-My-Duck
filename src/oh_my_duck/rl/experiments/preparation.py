import hashlib
import json
from pathlib import Path
import subprocess

REQUIRED = ('smoke', 'smoke-export', 'smoke-rehearsal', 'resume-check', 'capacity', 'capacity-export')
CRITICAL = ('src/oh_my_duck', 'environments', 'configs', 'pyproject.toml', 'omd.py')


def validate_critical_inputs(source, root):
    changed = subprocess.check_output(
        ['git', 'diff', '--name-only', source, '--', *CRITICAL], cwd=root, text=True)
    untracked = subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '--', *CRITICAL], cwd=root, text=True)
    if changed.strip() or untracked.strip():
        raise ValueError('Training/evaluation inputs changed: ' + changed + untracked)
    return changed.strip()


def validate_preparation(path, spec, root):
    path = Path(path)
    report = json.loads(path.read_text())
    if report['status'] != 'prepared' or report['spec'] != spec:
        raise ValueError('Preparation must be complete and match the entire training spec')
    if any(report['stages'].get(name, {}).get('status') != 'completed' for name in REQUIRED):
        raise ValueError('Preparation is missing a required successful gate')
    source = json.loads((path.parent.parent/'campaign.json').read_text())['source_commit']
    changed = validate_critical_inputs(source, root)
    for name in ('smoke-export', 'capacity-export'):
        if not (path.parent/name/'policy.onnx').is_file():
            raise ValueError('Prepared export missing: ' + name)
    return {'result': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_commit': source, 'critical_input_diff': changed.strip()}


def reuse_smoke(path, spec, root):
    """Reuse completed smoke gates from an explicitly interrupted size sweep."""
    path = Path(path)
    report = json.loads(path.read_text())
    previous = report['spec']
    keys = ('id', 'backend', 'framework', 'task', 'experiment', 'seed', 'learning_rate',
            'learning_rate_mode', 'critic_observations', 'initial_episode_phase', 'action_rate_delay_iterations', 'low_speed_tracking_boost')
    if any(previous.get(k) != spec.get(k) for k in keys):
        raise ValueError('Smoke task, recipe, framework or seed differs')
    required = ('smoke', 'smoke-export', 'smoke-rehearsal', 'resume-check')
    if any(report['stages'].get(k, {}).get('status') != 'completed' for k in required):
        raise ValueError('Cannot reuse incomplete smoke gates')
    command = report['stages']['smoke']['command']
    count_option, updates_option = ('--num-envs', '--iterations') if spec['framework'] == 'sb3' else ('--env.scene.num-envs', '--agent.max-iterations')
    if command[command.index(count_option)+1] != '64' or int(command[command.index(updates_option)+1]) < 5:
        raise ValueError('Reused smoke must have 64 environments and at least five updates')
    source = json.loads((path.parent.parent/'campaign.json').read_text())['source_commit']
    validate_critical_inputs(source, root)
    if not (path.parent/'smoke-export/policy.onnx').is_file() or not list((path.parent/'smoke-rehearsal').glob('*.mp4')):
        raise ValueError('Reused smoke export or CPU video missing')
    capacity_run = None
    for stage in ('capacity', f'scaling-{spec["num_envs"]}'):
        evidence = report['stages'].get(stage, {})
        command = evidence.get('command', [])
        if evidence.get('status') == 'completed' and count_option in command and int(command[command.index(count_option)+1]) == spec['num_envs']:
            capacity_run = evidence['run']
            break
    return {'result': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_commit': source, 'capacity_run': capacity_run,
            'reuse': 'completed_smoke_and_matching_capacity_only; interrupted_larger_cases_excluded'}
