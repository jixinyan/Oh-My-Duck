"""Task discovery and extension must work without importing simulator libraries."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from oh_my_duck.rl.training.tasks import ConfigRef, TaskBinding, TaskRegistry, TaskSpec, project_tasks

ROOT = Path(__file__).resolve().parents[1]


class TaskRegistryTests(unittest.TestCase):
    def test_discovery_is_lightweight_and_reports_only_explicit_bindings(self):
        registry = project_tasks(ROOT)
        self.assertEqual(len(registry.list()), 33)
        self.assertEqual(registry.list('isaac-newton'), [])
        for name in ('mujoco', 'torch', 'warp', 'isaaclab'):
            self.assertNotIn(name, sys.modules)

    def test_custom_task_id_is_not_tied_to_robot_name_or_dispatcher(self):
        registry = TaskRegistry()
        config = ConfigRef('example:config')
        registry.register(TaskSpec('MyBalance-v0', 'custom', {'custom-physics': TaskBinding(config, config, config)}))
        self.assertEqual(registry.list('custom-physics')[0].id, 'MyBalance-v0')
        with self.assertRaises(ValueError):
            registry.get('MyBalance-v0').binding('mujoco')
        with self.assertRaises(ValueError):
            registry.register(registry.get('MyBalance-v0'))

    def test_config_construction_never_mutates_prototypes(self):
        module = types.ModuleType('omd_test_recipe')
        module.prototype = {'actor': {'widths': [64, 32]}}
        module.factory = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {'omd_test_recipe': module}):
            ref = ConfigRef('omd_test_recipe:prototype')
            first = ref.build()
            first['actor']['widths'].append(16)
            self.assertEqual(ref.build()['actor']['widths'], [64, 32])
            factory = ConfigRef('omd_test_recipe:factory', {'rough': False})
            self.assertEqual(factory.build(play=True), {'rough': False, 'play': True})

    def test_callers_cannot_mutate_registered_bindings(self):
        registry = project_tasks(ROOT)
        task = registry.list()[0]
        task.bindings.clear()
        self.assertIn('mujoco', registry.get(task.id).bindings)
