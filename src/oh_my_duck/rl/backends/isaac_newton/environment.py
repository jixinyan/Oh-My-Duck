"""Runtime guards make alternate physics or reordered action dimensions fail explicitly."""
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from oh_my_duck.rl.backends.isaac_newton.contracts import JOINT_NAMES
from oh_my_duck.rl.backends.isaac_newton.paths import require_asset


class DiagnosticEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        # Base destructor can run even when our preflight rejects the config.
        self._is_closed = True
        self._capture_terminal_observations = False
        self._terminal_snapshot = None
        if not isinstance(cfg.sim.physics, NewtonCfg) or not isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("This task requires Newton with MJWarp; alternate physics is unsupported")
        if cfg.sim.physics.solver_cfg.use_mujoco_cpu:
            raise ValueError("This backend requires GPU MJWarp, not the CPU solver")
        if abs(cfg.sim.dt * cfg.decimation - 0.02) > 1e-10:
            raise ValueError("Policy control must remain at 50 Hz")
        action = cfg.actions.joint_pos
        if tuple(action.joint_names) != JOINT_NAMES or not action.preserve_order or action.scale != 1.0:
            raise ValueError("Canonical 14-joint order and action scale are required")
        require_asset()
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        # Capture final policy state before the base environment resets it. SB3 needs
        # this state for timeout bootstrapping; the ordinary step observation is post-reset.
        if self._capture_terminal_observations:
            state = self.observation_manager.compute(update_history=False)["policy"]
            self._terminal_snapshot = (env_ids.detach().clone(), state[env_ids].detach().clone())
        super()._reset_idx(env_ids)
