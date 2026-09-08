"""Reuse completed gates without repeating training or accepting partial evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

REQUIRED = ('smoke', 'smoke-export', 'smoke-rehearsal', 'resume-check', 'capacity', 'capacity-export')
CRITICAL = ('src/oh_my_duck/rl/tasks', 'src/oh_my_duck/rl/learners',
            'src/oh_my_duck/rl/backends', 'src/oh_my_duck/rl/evaluation',
            'src/oh_my_duck/rl/artifacts', 'src/oh_my_duck/rl/training',
            'src/oh_my_duck/robotics', 'environments', 'configs/upstream.json',
            'configs/training.json', 'pyproject.toml')


def validate_preparation(path, spec, root):
    path = Path(path)
    report = json.loads(path.read_text())
    if report['status'] != 'prepared' or report['spec'] != spec:
        raise ValueError('Preparation must be complete and match the entire training spec')
    if any(report['stages'].get(name, {}).get('status') != 'completed' for name in REQUIRED):
        raise ValueError('Preparation is missing a required successful gate')
    source = json.loads((path.parent.parent/'campaign.json').read_text())['source_commit']
    changed = subprocess.check_output(['git', 'diff', '--name-only', source, 'HEAD', '--', *CRITICAL], cwd=root, text=True)
    if changed.strip():
        raise ValueError('Training/evaluation inputs changed since preparation: ' + changed)
    for name in ('smoke-export', 'capacity-export'):
        if not (path.parent/name/'policy.onnx').is_file():
            raise ValueError('Prepared export missing: ' + name)
    return {'result': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_commit': source, 'critical_input_diff': changed.strip()}
