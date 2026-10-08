import asyncio
import math
import unittest

from jsonschema.exceptions import SchemaError

from oh_my_duck.agentic.skills.simulation import SimulationSkillRunner
from oh_my_duck.agentic.tools import ToolSchemaError, validate_tool_arguments
from oh_my_duck.agentic.tools.simulation import simulation_tools
from oh_my_duck.robotics.backends.simulation import SimulationBackend


class ToolSchemaTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulationBackend(
            robot_id="schema-check", backend="cpu-mujoco-bam", policy_path=None, task_id="schema-check"
        )
        self.runner = SimulationSkillRunner(self.backend)
        self.catalog = simulation_tools(self.backend, self.runner)

    def test_actual_capability_handler_preserves_identity(self):
        result = asyncio.run(self.catalog.invoke("get_capabilities", "actual-capabilities", {}))
        self.assertEqual(result.request_id, "actual-capabilities")
        self.assertEqual(result.payload["identity"]["robot_id"], self.backend.robot_id)
        self.assertEqual(result.payload["backend"], self.backend.backend)
        self.assertEqual(result.payload["command_limits"]["vx_m_s"], [-0.5, 0.5])

    def test_declared_tool_arguments_are_valid(self):
        arguments = {
            "get_capabilities": {}, "get_robot_state": {}, "stop_motion": {},
            "read_sensor": {"sensor_id": "imu", "max_age_ms": 100},
            "run_skill": {"skill_id": "move_for", "parameters": {
                "vx_m_s": 0.1, "vy_m_s": 0, "yaw_rad_s": 0, "duration_s": 0.2,
            }},
            "get_task_status": {"task_id": "task-id"},
            "cancel_task": {"task_id": "task-id", "reason": "user requested stop"},
        }
        for definition in self.catalog.definitions():
            with self.subTest(tool=definition.name):
                validate_tool_arguments(arguments[definition.name], definition.parameter_schema)

    def test_invalid_arguments_stop_before_uninitialized_physics(self):
        cases = (
            ("get_robot_state", {"extra": True}),
            ("read_sensor", {"sensor_id": "imu"}),
            ("read_sensor", {"sensor_id": "imu", "max_age_ms": -1}),
            ("read_sensor", {"sensor_id": "imu", "max_age_ms": True}),
            ("read_sensor", {"sensor_id": 1, "max_age_ms": 0}),
            ("stop_motion", {"extra": True}),
            ("get_task_status", {"task_id": ""}),
            ("cancel_task", {"task_id": "task-id", "reason": ""}),
            ("run_skill", {"skill_id": "walk", "parameters": {}}),
            ("run_skill", {"skill_id": "move_for", "parameters": {
                "vx_m_s": 0.6, "vy_m_s": 0, "yaw_rad_s": 0, "duration_s": 0.2,
            }}),
            ("run_skill", {"skill_id": "move_for", "parameters": {
                "vx_m_s": 0, "vy_m_s": 0, "yaw_rad_s": 0, "duration_s": 0,
            }}),
        )
        for name, arguments in cases:
            with self.subTest(tool=name, arguments=arguments), self.assertRaises(ToolSchemaError):
                asyncio.run(self.catalog.invoke(name, "invalid-arguments", arguments))
        self.assertEqual(self.backend._sequence, 0)
        self.assertIsNone(self.backend._measurements)
        self.assertIsNone(self.runner._active)
        self.assertEqual(self.runner._jobs, {})

    def test_standard_schema_references_and_conditional_requirements(self):
        schema = {
            "type": "object", "additionalProperties": False,
            "$defs": {"distance": {"type": "number", "exclusiveMinimum": 0}},
            "required": ["mode"], "properties": {
                "mode": {"enum": ["move", "observe"]},
                "distance_m": {"$ref": "#/$defs/distance"},
            },
            "if": {"properties": {"mode": {"const": "move"}}},
            "then": {"required": ["distance_m"]},
            "else": {"properties": {"distance_m": False}},
        }
        validate_tool_arguments({"mode": "move", "distance_m": 0.2}, schema)
        validate_tool_arguments({"mode": "observe"}, schema)
        for arguments in ({"mode": "move"}, {"mode": "move", "distance_m": 0},
                          {"mode": "observe", "distance_m": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(ToolSchemaError):
                validate_tool_arguments(arguments, schema)

    def test_standard_schema_array_and_dependency_rules(self):
        schema = {"type": "object", "properties": {
            "position": {"type": "array", "prefixItems": [
                {"type": "number"}, {"type": "number"},
            ], "items": False, "minItems": 2},
            "labels": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "frame": {"type": "string"},
        }, "dependentRequired": {"position": ["frame"]}}
        validate_tool_arguments({"position": [1, 2], "frame": "world", "labels": ["chair", "table"]}, schema)
        for arguments in ({"position": [1, 2]}, {"position": [1, 2, 3], "frame": "world"},
                          {"position": [True, 2], "frame": "world"}, {"labels": ["chair", "chair"]}):
            with self.subTest(arguments=arguments), self.assertRaises(ToolSchemaError):
                validate_tool_arguments(arguments, schema)

    def test_invalid_schema_and_nonfinite_numbers_fail(self):
        with self.assertRaises(SchemaError):
            validate_tool_arguments({}, {"type": "invented"})
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_tool_arguments({"value": value}, {"type": "object"})


if __name__ == "__main__":
    unittest.main()
