import json
import os
from pathlib import Path
import subprocess
import sys

from jsonschema import ValidationError
import pytest

from oh_my_duck.integrations.edh.scene_configuration import validate_scene_configuration


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "configs/simulation-demo"


@pytest.mark.parametrize("path", sorted(SCENES.glob("*.json")))
def test_declared_scenes_use_the_native_configuration_schema(path):
    validate_scene_configuration(json.loads(path.read_text()))


@pytest.mark.parametrize("fields, value", [
    (("backend",), "physx"), (("scene_id",), "../apartment"),
    (("robot_model",), "groundcontact_rollers"), (("device",), "cuda:0"),
    (("usd_path",), "scene.usd"), (("observer_renderer",), "isaac_rtx"),
    (("task_instruction",), "   "), (("spawn_pose", "x_m"), True),
    (("spawn_pose", "z_m"), 0.125), (("goal", "kind"), "point"),
    (("goal", "room"), "unknown"), (("goal", "hold_ticks"), True),
    (("goal", "hold_ticks"), 501), (("budget", "max_control_steps"), 4),
    (("budget", "max_control_steps"), 9007199254740992),
    (("budget", "max_wall_time_s"), 0),
])
def test_invalid_cpu_configuration_is_rejected(fields, value):
    configuration = json.loads((SCENES / "apartment-metric.json").read_text())
    target = configuration
    for name in fields[:-1]:
        target = target[name]
    target[fields[-1]] = value
    with pytest.raises(ValidationError):
        validate_scene_configuration(configuration)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scene_numbers_fail_before_schema_validation(value):
    configuration = json.loads((SCENES / "office.json").read_text())
    configuration["spawn_pose"]["x_m"] = value
    with pytest.raises(ValueError, match="Out of range float values"):
        validate_scene_configuration(configuration)


def test_cpu_scene_rejects_cuda_assignment_before_sdk_loading():
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMD_PROJECT_ROOT": str(ROOT),
                   "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-m", "oh_my_duck", "harness", "--scene-config",
                             str(SCENES / "apartment-metric.json"), "--worker-cuda-device", "2"],
                            cwd=Path.home(), env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "CPU apartment scenes require no CUDA device assignment" in result.stderr
    assert "FileNotFoundError" not in result.stderr
