import json
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_scene_configuration(configuration: dict) -> None:
    json.dumps(configuration, allow_nan=False)
    schema = json.loads(Path(__file__).with_name("scene.schema.json").read_text())
    Draft202012Validator(schema).validate(configuration)
