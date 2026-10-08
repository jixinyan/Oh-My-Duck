import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments, message", [
    (["--worker-policy-registry", "/weights/registry.json"], "requires --worker-host"),
    (["--worker-policy-registry", "relative.json", "--worker-host", "jd_B300"], "absolute path"),
    (["--worker-policy-registry", "/weights/../registry.json", "--worker-host", "jd_B300"], "parent components"),
    (["--policy-registry", "local.json", "--worker-host", "jd_B300"], "--worker-policy-registry"),
    (["--policy-registry", "local.json", "--worker-policy-registry", "/weights/registry.json"], "local or remote"),
])
def test_invalid_registry_transport_arguments_fail_before_sdk_or_provider_loading(arguments, message):
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMD_PROJECT_ROOT": str(ROOT),
                   "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-m", "oh_my_duck", "harness", *arguments],
                            cwd=Path.home(), env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert message in result.stderr
    assert "FileNotFoundError" not in result.stderr
