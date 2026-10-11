import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_my_duck.rl.training.arguments import task_selection
from oh_my_duck.rl.training.isaac_newton import DIAGNOSTIC_TASK, is_diagnostic


class TrainingArgumentTests(unittest.TestCase):
    def test_task_option_forms_preserve_native_arguments(self):
        task = "Mjlab-StandUp-Flat-MicroDuck"
        for arguments in (
            ["--task", task, "--policy", "policy.onnx"],
            ["--task=" + task, "--policy=policy.onnx"],
            ["--output", "episode", "--task", task, "--help"],
        ):
            original = arguments.copy()
            with self.subTest(arguments=arguments):
                self.assertEqual(task_selection(arguments), task)
                self.assertEqual(arguments, original)

    def test_diagnostic_selection_requires_explicit_task_option(self):
        task = "Mjlab-Velocity-Flat-MicroDuck"
        for arguments in (
            [task, "--agent.run-name", DIAGNOSTIC_TASK],
            [task, "--output", DIAGNOSTIC_TASK],
            [task, "--description", DIAGNOSTIC_TASK],
            ["--task-extra=" + DIAGNOSTIC_TASK],
            ["--", "--task", DIAGNOSTIC_TASK],
        ):
            with self.subTest(arguments=arguments):
                self.assertFalse(is_diagnostic(arguments))
        for arguments in (["--task", DIAGNOSTIC_TASK], ["--task=" + DIAGNOSTIC_TASK]):
            with self.subTest(arguments=arguments):
                self.assertTrue(is_diagnostic(arguments))

    def test_invalid_task_options_fail_at_argument_reading(self):
        for arguments in (
            ["--task"],
            ["--task", "first", "--task=second"],
            ["--task", DIAGNOSTIC_TASK, "--task=" + DIAGNOSTIC_TASK],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    task_selection(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
