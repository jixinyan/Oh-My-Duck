"""Framework routing must preserve physics selection and reject unsupported pairs."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from oh_my_duck.training.base import BackendCommand
from oh_my_duck.training.frameworks import FrameworkBinding, RLFrameworkRegistry, default_framework_registry
from oh_my_duck.training.registry import BackendUnavailable, default_registry
from oh_my_duck.training.isaac_newton import DIAGNOSTIC_TASK


class FrameworkTests(unittest.TestCase):
    def test_rsl_route_preserves_existing_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / '.envs/mujoco/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            args = ['Mjlab-Velocity-Flat-MicroDuck', '--agent.max-iterations', '5']
            expected = default_registry().get('mujoco', root).command('train', args)
            actual = default_framework_registry().command('rsl-rl', 'mujoco', root, 'train', args)
            self.assertEqual(actual, expected)

    def test_unsupported_combinations_never_fall_back(self):
        registry = default_framework_registry()
        for framework, backend, op in [('unknown', 'mujoco', 'train'), ('sb3', 'mujoco', 'unsupported'),
                                        ('sb3', 'isaac-newton', 'export'), ('rsl-rl', 'unknown', 'train')]:
            with self.subTest(framework=framework, backend=backend, operation=op):
                with self.assertRaises(BackendUnavailable):
                    registry.command(framework, backend, ROOT, op, [])

    def test_sb3_uses_same_environment_with_explicit_diagnostic_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / '.envs/isaac-newton/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            registry = default_framework_registry()
            with self.assertRaises(BackendUnavailable):
                registry.command('sb3', 'isaac-newton', root, 'train', [])
            args = ['--task', DIAGNOSTIC_TASK]
            sb3 = registry.command('sb3', 'isaac-newton', root, 'train', args)
            rsl = registry.command('rsl-rl', 'isaac-newton', root, 'train', args)
            self.assertEqual(sb3.argv[:3], (str(interpreter), '-m', 'omd_isaac.sb3_train'))
            self.assertEqual(sb3.environment, rsl.environment)
            self.assertEqual(sb3.cwd, rsl.cwd)
            self.assertNotIn('stable_baselines3', sys.modules)

    def test_new_framework_can_register_without_changing_dispatch(self):
        registry = RLFrameworkRegistry()
        expected = BackendCommand(('custom-trainer',), ROOT, {})
        binding = FrameworkBinding('mujoco', ('train',), 'test_only', lambda *args: expected)
        registry.register('custom', 'Custom framework', [binding])
        self.assertEqual(registry.command('custom', 'mujoco', ROOT, 'train', []), expected)
        with self.assertRaises(ValueError):
            registry.register('custom', 'Duplicate', [binding])
        with self.assertRaises(ValueError):
            registry.register('other', 'Duplicate binding', [binding, binding])

if __name__ == '__main__':
    unittest.main()
