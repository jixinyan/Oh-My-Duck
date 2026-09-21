import json
import subprocess

import pytest

from oh_my_duck.rl.experiments.preparation import validate_critical_inputs


def git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def write_input(root, name, revision):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'revision': revision}))
    return path


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init', '--quiet')
    write_input(tmp_path, 'src/oh_my_duck/rl/mdp/observations.json', 1)
    write_input(tmp_path, 'docs/notes.json', 1)
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '--quiet', '-m', 'Record input data')
    return tmp_path, git(tmp_path, 'rev-parse', 'HEAD')


@pytest.mark.parametrize('name', [
    'src/oh_my_duck/rl/mdp/observations.json',
    'src/oh_my_duck/rl/mdp/rewards.json',
    'src/oh_my_duck/infrastructure/provenance.json',
    'src/oh_my_duck/core/paths.json',
    'src/oh_my_duck/rl/experiments/options.json',
    'src/oh_my_duck/robotics/parameters.json',
    'configs/training.json',
    'environments/dependencies.json',
])
def test_committed_input_changes_require_new_gates(repository, name):
    root, source = repository
    write_input(root, name, 2)
    git(root, 'add', '.')
    git(root, 'commit', '--quiet', '-m', 'Update input data')
    with pytest.raises(ValueError, match='inputs changed') as error:
        validate_critical_inputs(source, root)
    assert name in str(error.value)


@pytest.mark.parametrize('state', ['modified', 'staged', 'untracked', 'deleted'])
def test_uncommitted_input_changes_require_new_gates(repository, state):
    root, source = repository
    name = 'src/oh_my_duck/rl/mdp/observations.json'
    if state == 'deleted':
        (root / name).unlink()
    else:
        if state == 'untracked':
            name = 'src/oh_my_duck/rl/mdp/rewards.json'
        write_input(root, name, 2)
        if state == 'staged':
            git(root, 'add', name)
    with pytest.raises(ValueError, match='inputs changed') as error:
        validate_critical_inputs(source, root)
    assert name in str(error.value)


def test_unchanged_inputs_and_documentation_changes_allow_reuse(repository):
    root, source = repository
    assert validate_critical_inputs(source, root) == ''
    write_input(root, 'docs/notes.json', 2)
    git(root, 'add', '.')
    git(root, 'commit', '--quiet', '-m', 'Update documentation data')
    write_input(root, 'docs/new-notes.json', 1)
    assert validate_critical_inputs(source, root) == ''


def test_unknown_source_commit_fails(repository):
    root, _ = repository
    with pytest.raises(subprocess.CalledProcessError):
        validate_critical_inputs('unknown-source', root)
