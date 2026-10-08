import json
from pathlib import Path
from typing import Mapping

from jsonschema import Draft202012Validator

from oh_my_duck.perception.client import PerceptionClient


def perception_sources(configuration: Mapping[str, object]) -> tuple[str, ...]:
    if "perception_endpoint" in configuration:
        PerceptionClient(configuration["perception_endpoint"])
        return ("simulator_ground_truth", "models")
    return ("simulator_ground_truth",)


def validate_scene_configuration(configuration: dict) -> None:
    json.dumps(configuration, allow_nan=False)
    schema = json.loads(Path(__file__).with_name("scene.schema.json").read_text())
    Draft202012Validator(schema).validate(configuration)
    perception_sources(configuration)
