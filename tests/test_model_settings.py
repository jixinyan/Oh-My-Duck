import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from oh_my_duck.integrations.model_settings import (
    add_model_arguments, resolve_model_settings, validate_model_options,
)
from oh_my_duck.validation.release.campaign import gpu_query_timeout


ROOT = Path(__file__).resolve().parents[1]


def test_verified_native_model_settings_are_preserved():
    recorded = json.loads((ROOT / 'configs/project.json').read_text())['release_readiness']['native_task_context']
    settings = resolve_model_settings(recorded['model'], 'responses',
        reasoning_effort=recorded['reasoning_effort'], max_output_tokens=recorded['max_output_tokens'])
    assert settings.model == 'gpt-6-luna' and settings.max_output_tokens == 8192
    parser = argparse.ArgumentParser()
    add_model_arguments(parser, max_output_tokens=8192, reasoning_effort='high')
    submitted = parser.parse_args(settings.arguments())
    assert vars(submitted) == settings.metadata()
    assert set(settings.metadata()) == {'model', 'model_api', 'reasoning_effort', 'max_output_tokens'}


@pytest.mark.parametrize('budget', [0, 255, 8193, True, 4096.0, float('nan'), float('inf'), None])
def test_invalid_budget_fails_at_model_boundary(budget):
    with pytest.raises(ValueError, match='token 预算'):
        validate_model_options(None, None, None, budget)


@pytest.mark.parametrize('model', ['', ' ', 'model\n', 'model\x00', True, 42, [], {}])
def test_invalid_model_identity_fails_at_model_boundary(model):
    with pytest.raises(ValueError, match='模型标识'):
        resolve_model_settings(model, 'responses')


@pytest.mark.parametrize('wire_api', ['', 'unknown', None, True])
def test_unknown_provider_api_is_rejected(wire_api):
    with pytest.raises(ValueError, match='wire_api'):
        resolve_model_settings('gpt-6-luna', wire_api)


@pytest.mark.parametrize('api', ['chat', '', 'unknown', True])
def test_invalid_explicit_api_is_rejected(api):
    with pytest.raises(ValueError, match='Model API'):
        resolve_model_settings('gpt-6-luna', 'responses', model_api=api)


@pytest.mark.parametrize('effort', ['', 'unknown', True])
def test_invalid_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValueError, match='reasoning effort'):
        resolve_model_settings('gpt-6-luna', 'responses', reasoning_effort=effort)


def test_declared_provider_chat_alias_and_explicit_model_selection():
    assert resolve_model_settings('gpt-6-luna', 'chat').model_api == 'chat-completions'
    selected = resolve_model_settings('gpt-6-luna', 'chat', model='gpt-6.1-sol', model_api='responses',
                                       reasoning_effort='high', max_output_tokens=8192)
    assert selected.model == 'gpt-6.1-sol' and selected.model_api == 'responses'


def test_elapsed_gpu_deadline_fails_before_query():
    with pytest.raises(TimeoutError, match='三十秒期限'):
        gpu_query_timeout(time.monotonic() - 1)
    deadline = time.monotonic() + 30
    assert 0 < gpu_query_timeout(deadline) <= 30


@pytest.mark.parametrize('budget', [0, 255, 8193, 32768])
@pytest.mark.parametrize('module', ['oh_my_duck.validation.release.campaign', 'oh_my_duck.validation.release.worker'])
def test_release_budget_rejected_before_sources_or_output(tmp_path, module, budget):
    output = tmp_path / 'acceptance'
    if module.endswith('campaign'):
        arguments = ['--worker-host', 'jd_B300', '--worker-root', '/unavailable/source',
            '--worker-python', '/unavailable/python', '--worker-edh-source', '/unavailable/edh',
            '--worker-policy-dir', '/unavailable/policies', '--edh-source', str(tmp_path / 'edh'),
            '--remote-provider-config', '/unavailable/provider.toml', '--preflight-only']
    else:
        arguments = ['--plan', str(tmp_path / 'plan.json'), '--catalog', str(tmp_path / 'policies'),
            '--edh-source', str(tmp_path / 'edh'), '--harness-manifest', str(tmp_path / 'manifest.json'),
            '--provider-config', str(tmp_path / 'provider.toml')]
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-m', module, *arguments, '--output', str(output),
                             '--max-output-tokens', str(budget)], cwd=Path.home(), env=environment,
                            check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and '模型输出 token 预算必须介于 256 和 8192 之间' in result.stderr
    assert 'FileNotFoundError' not in result.stderr and 'KeyError' not in result.stderr
    assert not output.exists()
