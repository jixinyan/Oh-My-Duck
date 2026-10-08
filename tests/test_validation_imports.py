import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "oh_my_duck.cli.validation",
    "oh_my_duck.validation.release.plans",
    "oh_my_duck.validation.harness.control",
    "oh_my_duck.validation.harness.pose",
    "oh_my_duck.validation.harness.pose_records",
    "oh_my_duck.experience.harness_replay",
    "oh_my_duck.infrastructure.usd_runtime",
    "oh_my_duck.validation.release.assets",
    "oh_my_duck.rl.training.runtime",
    "oh_my_duck.rl.evaluation.task",
    "oh_my_duck.rl.backends.isaac_newton.task_binding.collision_assets",
    "oh_my_duck.rl.backends.isaac_newton.task_binding.collision_model",
    "oh_my_duck.rl.backends.isaac_newton.task_binding.contact_model",
])
def test_metadata_and_transport_imports_leave_execution_dependencies_unloaded(module):
    code = "import importlib,json,sys; importlib.import_module(sys.argv[1]); print(json.dumps(sorted(sys.modules)))"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", code, module], cwd=Path.home(), env=environment,
                            check=True, capture_output=True, text=True, timeout=30)
    loaded = {name.partition(".")[0] for name in json.loads(result.stdout)}
    assert not loaded.intersection({"torch", "mujoco", "numpy", "onnxruntime", "isaaclab", "isaacsim",
                                    "newton", "warp", "pxr", "physical_harness"})


def test_validation_help_is_available_outside_a_checkout():
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("OMD_PROJECT_ROOT", None)
    result = subprocess.run([sys.executable, "-m", "oh_my_duck", "validate", "--help"],
                            cwd=Path.home(), env=environment, check=True, capture_output=True,
                            text=True, timeout=30)
    assert "release-plan" in result.stdout
