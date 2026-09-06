import copy
import json
from pathlib import Path
import unittest

from training.common.protocol import compile_schedule


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/eval/flat_walk.json").read_text())

    def test_boundary_changes_happen_on_exact_ticks(self):
        schedule = compile_schedule(self.config)
        self.assertEqual(len(schedule), 700)
        self.assertEqual(schedule[99][2], (0, 0, 0))
        self.assertEqual(schedule[100][2], (0.1, 0, 0))
        self.assertEqual(schedule[300][2], (0, 0, 0))
        self.assertEqual(schedule[400][2], (0, 0, 0.5))
        self.assertEqual(schedule[600][2], (0, 0, 0))

    def test_rejects_ambiguous_timing_and_nonfinite_commands(self):
        for key, value in [("duration_s", 0.021), ("duration_s", -1), ("twist", [float("nan"), 0, 0])]:
            config = copy.deepcopy(self.config)
            config["segments"][0][key] = value
            with self.assertRaises(ValueError):
                compile_schedule(config)

    def test_control_frequency_is_part_of_the_contract(self):
        self.config["decimation"] = 5
        with self.assertRaises(ValueError):
            compile_schedule(self.config)


if __name__ == "__main__":
    unittest.main()
