"""CPU-only checks against the actual pinned Isaac packages; no simulation startup."""
import gc
import sys
import unittest
from unittest.mock import patch
from oh_my_duck.rl.backends.isaac_newton.config import DiagnosticEnvCfg
from oh_my_duck.rl.backends.isaac_newton.environment import DiagnosticEnv
from oh_my_duck.rl.backends.isaac_newton.tasks import TASK_ID, register_tasks
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg


class InstalledConfigTests(unittest.TestCase):
    def test_registered_task_and_config_are_constructible_without_assets(self):
        import gymnasium as gym
        register_tasks()
        self.assertEqual(gym.spec(TASK_ID).entry_point, "oh_my_duck.rl.backends.isaac_newton.environment:DiagnosticEnv")
        cfg = DiagnosticEnvCfg()
        cfg.validate()
        self.assertIsInstance(cfg.sim.physics, NewtonCfg)
        self.assertIsInstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
        camera = CameraCfg(prim_path="/World/Camera", width=320, height=240, data_types=["rgb"],
            spawn=PinholeCameraCfg(), renderer_cfg=NewtonWarpRendererCfg())
        self.assertIsInstance(camera.renderer_cfg, NewtonWarpRendererCfg)

    def test_alternate_engine_is_rejected_before_asset_or_simulator_startup(self):
        cfg = DiagnosticEnvCfg()
        cfg.sim.physics = object()
        with self.assertRaisesRegex(ValueError, "requires Newton"):
            DiagnosticEnv(cfg)
        cfg = DiagnosticEnvCfg()
        cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
        with self.assertRaisesRegex(ValueError, "requires GPU"):
            DiagnosticEnv(cfg)

    def test_rejected_configuration_does_not_raise_during_cleanup(self):
        errors = []
        with patch.object(sys, "unraisablehook", errors.append):
            cfg = DiagnosticEnvCfg()
            cfg.sim.physics = object()
            with self.assertRaises(ValueError):
                DiagnosticEnv(cfg)
            gc.collect()
        self.assertEqual(errors, [])

    def test_control_and_order_overrides_are_rejected_before_startup(self):
        cfg = DiagnosticEnvCfg()
        cfg.decimation = 2
        with self.assertRaisesRegex(ValueError, "50 Hz"):
            DiagnosticEnv(cfg)
        cfg = DiagnosticEnvCfg()
        cfg.actions.joint_pos.joint_names.reverse()
        with self.assertRaisesRegex(ValueError, "Canonical"):
            DiagnosticEnv(cfg)

if __name__ == "__main__":
    unittest.main()
