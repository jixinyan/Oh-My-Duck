import json
from pathlib import Path
import shutil
from uuid import uuid4

from jsonschema import ValidationError
import pytest

from release_plan import validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiments/runtime-release-acceptance.json"


@pytest.fixture
def inputs():
    directory = ROOT / ".cache/release-plan-tests" / uuid4().hex
    directory.mkdir(parents=True)
    yield directory, json.loads(PLAN.read_text())
    shutil.rmtree(directory)


def save(directory, data):
    path = directory / "plan.json"
    path.write_text(json.dumps(data) + "\n")
    return path


def test_current_release_plan_preserves_all_stage_inputs():
    plan, scenes, hashes = validate_plan(ROOT, PLAN)
    assert len(plan["stages"]) == 6
    assert len(scenes) == 4
    assert len(hashes) == 9
    assert all(len(value) == 64 for value in hashes.values())


def test_duplicate_stage_id_is_rejected(inputs):
    directory, plan = inputs
    plan["stages"][1]["id"] = plan["stages"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        validate_plan(ROOT, save(directory, plan))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_navigation_distance_is_rejected(inputs, value):
    directory, plan = inputs
    plan["stages"][-1]["minimum_distance_m"] = value
    with pytest.raises((ValueError, ValidationError)):
        validate_plan(ROOT, save(directory, plan))


def test_unknown_metric_suite_is_rejected(inputs):
    directory, plan = inputs
    plan["stages"][0]["suite"] = "missing-suite"
    with pytest.raises(ValueError, match="Unknown metric suite"):
        validate_plan(ROOT, save(directory, plan))


@pytest.mark.parametrize("name", ["../README.md", str(ROOT / "README.md")])
def test_input_paths_require_project_relative_files(inputs, name):
    directory, plan = inputs
    plan["stages"][-1]["instruction"] = name
    with pytest.raises(ValueError, match="relative"):
        validate_plan(ROOT, save(directory, plan))


@pytest.mark.parametrize("field,value", [
    ("backend", "physx"), ("robot_model", "unknown-model"), ("device", "cuda:1"),
    ("observer_renderer", "unknown-renderer"), ("usd_path", "../scene.usd"),
    ("spawn_pose", {"x_m": 1, "y_m": 2, "z_m": 0.125, "yaw_rad": float("nan")}),
    ("budget", {"max_control_steps": True, "max_wall_time_s": 2400}),
])
def test_scene_errors_fail_before_runtime_launch(inputs, field, value):
    directory, plan = inputs
    stage = plan["stages"][-1]
    configuration = json.loads((ROOT / stage["scene"]).read_text())
    configuration[field] = value
    path = directory / "scene.json"
    path.write_text(json.dumps(configuration) + "\n")
    stage["scene"] = str(path.relative_to(ROOT))
    with pytest.raises((ValueError, ValidationError)):
        validate_plan(ROOT, save(directory, plan))
