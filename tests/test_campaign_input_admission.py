import json
from pathlib import Path
import subprocess
import sys

import pytest

from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.experiments.campaign import assigned_devices, load_plan


ROOT = project_root()
PLAN = ROOT / 'configs/experiments/representative-8192.json'
SEARCH_PLAN = ROOT / 'configs/experiments/representative-throughput.json'


@pytest.mark.parametrize('scope,field', [
    ('plan', 'smoke_iterations'), ('plan', 'checkpoint_interval'),
    ('run', 'num_envs'), ('run', 'iterations'), ('run', 'seed'),
    ('search', 'warmup_updates'), ('search', 'updates'),
])
@pytest.mark.parametrize('value', [True, False, 64.5, '64', None, float('inf'), float('nan')])
def test_native_integer_fields_reject_other_types(tmp_path, scope, field, value):
    plan = json.loads((SEARCH_PLAN if scope == 'search' else PLAN).read_text())
    target = {'plan': plan, 'run': plan['runs'][0],
              'search': plan.get('environment_search')}[scope]
    target[field] = value
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=field):
        load_plan(path)


@pytest.mark.parametrize('value', [True, 1.0, '1', None])
def test_schema_version_requires_integer_one(tmp_path, value):
    plan = json.loads(PLAN.read_text())
    plan['schema_version'] = value
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='schema-1'):
        load_plan(path)


@pytest.mark.parametrize('field', ['memory_fraction', 'stop_below_best_fraction'])
@pytest.mark.parametrize('value', [True, False, '0.85', None, 0, -0.1, 1.1, float('inf'), float('nan')])
def test_search_fractions_require_finite_numbers(tmp_path, field, value):
    plan = json.loads(SEARCH_PLAN.read_text())
    plan['environment_search'][field] = value
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=field):
        load_plan(path)


@pytest.mark.parametrize('value', [None, False, [], 'search'])
def test_declared_search_requires_object(tmp_path, value):
    plan = json.loads(SEARCH_PLAN.read_text())
    plan['environment_search'] = value
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='environment_search'):
        load_plan(path)


@pytest.mark.parametrize('option', ['count', 'runs_per_gpu'])
@pytest.mark.parametrize('value', [True, False, 1.5, 0, -1, '1', None])
def test_assignment_counts_require_positive_integers(option, value):
    arguments = {'count': 1, 'runs_per_gpu': 1, option: value}
    with pytest.raises(ValueError, match='integer'):
        assigned_devices('2', **arguments)


def test_committed_campaigns_preserve_their_complete_configuration():
    checked = []
    for path in sorted((ROOT / 'configs/experiments').glob('*.json')):
        declared = json.loads(path.read_text())
        if 'smoke_iterations' in declared:
            assert load_plan(path) == declared
            checked.append(path)
    assert PLAN in checked and SEARCH_PLAN in checked and len(checked) >= 8


def test_actual_campaign_cli_rejects_counts_before_output_creation(tmp_path):
    plan = json.loads(PLAN.read_text())
    plan['runs'][0]['num_envs'] = True
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    output = tmp_path / 'campaign'
    result = subprocess.run([
        sys.executable, '-m', 'oh_my_duck.rl.experiments.campaign',
        '--config', str(path), '--output', str(output), '--dry-run',
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'num_envs must be an integer' in result.stderr
    assert not output.exists()


def test_actual_campaign_cli_preserves_all_eight_runs_without_launching(tmp_path):
    output = tmp_path / 'campaign'
    result = subprocess.run([
        sys.executable, '-m', 'oh_my_duck.rl.experiments.campaign',
        '--config', str(PLAN), '--output', str(output), '--dry-run',
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual['runs'] == json.loads(PLAN.read_text())['runs']
    assert actual['required_gpus'] == 8 and actual['runs_per_gpu'] == 1
    assert not output.exists()
