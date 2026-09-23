import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('existing_path', ['', 'additional-modules'])
def test_source_entrypoint_exposes_package_to_child_process(tmp_path, existing_path):
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    if existing_path:
        environment['PYTHONPATH'] = existing_path
    code = (
        'import os, runpy, subprocess, sys; '
        'runpy.run_path(sys.argv[1]); '
        'print(os.environ["PYTHONPATH"], flush=True); '
        'subprocess.run([sys.executable, "-S", "-m", '
        '"oh_my_duck.infrastructure.bootstrap_isaac", "--help"], check=True)'
    )
    result = subprocess.run([sys.executable, '-S', '-c', code, str(root / 'omd.py')],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, check=True)
    expected = str(root / 'src') + (os.pathsep + existing_path if existing_path else '')
    assert result.stdout.splitlines()[0] == expected
    assert '--rl-framework' in result.stdout
