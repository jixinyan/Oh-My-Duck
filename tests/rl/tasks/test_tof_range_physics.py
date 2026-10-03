import mujoco
import numpy as np
import unittest

from oh_my_duck.robotics.microduck.sim_sensors import tof_frame


class TofRangePhysicsTests(unittest.TestCase):
    def test_noisy_actual_surface_measurements_stay_in_sensor_range(self):
        for maximum in (2.0, 4.0):
            with self.subTest(maximum=maximum):
                self.check_surface_range(maximum)

    def check_surface_range(self, maximum):
        model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
  <site name="tof" pos="0 0 0"/>
  <geom type="box" pos="{maximum + 0.05} 0 0" size="0.05 1 1"/>
</worldbody></mujoco>
""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        generator = np.random.default_rng(20261003)
        measurements = []
        for _ in range(100):
            distances, statuses = tof_frame(model, data, model.site("tof").id,
                                            directions=np.array([[1.0, 0.0, 0.0]]),
                                            max_range_m=maximum, rng=generator)
            self.assertEqual(statuses, [5])
            self.assertGreaterEqual(distances[0], 0)
            self.assertLessEqual(distances[0], int(maximum * 1000))
            measurements.extend(distances)
        self.assertEqual(max(measurements), int(maximum * 1000))
        self.assertLess(min(measurements), int(maximum * 1000))

    def test_actual_surface_beyond_sensor_range_reports_no_target(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
  <site name="tof" pos="0 0 0"/>
  <geom type="box" pos="4.2 0 0" size="0.05 1 1"/>
</worldbody></mujoco>
""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        distances, statuses = tof_frame(model, data, model.site("tof").id,
                                        directions=np.array([[1.0, 0.0, 0.0]]),
                                        rng=np.random.default_rng(20261003))
        self.assertEqual(distances, [0])
        self.assertEqual(statuses, [255])


if __name__ == "__main__":
    unittest.main()
