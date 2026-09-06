"""Official task managers on Isaac Lab's native Newton physics lifecycle.

Managers retain their original order, delays, noise, reward functions, curriculum
and reset behavior. The constructor intentionally replaces only physics and scene
creation from mjlab; no MuJoCo simulation environment is instantiated here.
"""
import torch
from mjlab.envs import ManagerBasedRlEnv
from oh_my_duck.rl.backends.isaac_newton.task_binding.simulation import NewtonSimulation
from oh_my_duck.rl.backends.isaac_newton.task_binding.scene import NewtonScene


class NewtonTaskEnvironment(ManagerBasedRlEnv):
    def __init__(self,cfg,*,task,device,render_mode=None):
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from isaaclab_tasks.utils import launch_simulation
        from isaaclab.sim import SimulationCfg
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        from oh_my_duck.rl.backends.isaac_newton.config import SceneCfg
        from oh_my_duck.rl.backends.isaac_newton.bam_actuator import OfficialBamActuatorCfg
        from oh_my_duck.rl.backends.isaac_newton.task_binding.collisions import configure_scene
        from oh_my_duck.rl.backends.isaac_newton.paths import require_asset
        from oh_my_duck.rl.backends.isaac_newton.contracts import JOINT_NAMES
        if render_mode is not None:
            raise ValueError('Use the headless policy evaluator for video')
        if set(cfg.scene.entities) != {'robot'} or cfg.scene.terrain is None or cfg.scene.terrain.terrain_type != 'plane':
            raise ValueError('Only the explicit flat robot scene is currently bound')
        self.cfg=cfg
        if cfg.seed is not None:self.cfg.seed=self.seed(cfg.seed)
        scene_cfg=SceneCfg(num_envs=cfg.scene.num_envs,env_spacing=cfg.scene.env_spacing)
        scene_cfg.robot.spawn.usd_path=str(require_asset(task.model))
        configure_scene(scene_cfg,task.model)
        scene_cfg.robot.actuators={'official_bam':OfficialBamActuatorCfg(joint_names_expr=list(JOINT_NAMES))}
        native_cfg=ManagerBasedEnvCfg(scene=scene_cfg,decimation=1,actions={},observations={},events={},seed=cfg.seed,
            sim=SimulationCfg(device=device,dt=cfg.sim.mujoco.timestep,render_interval=cfg.decimation,
                physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=cfg.sim.mujoco.iterations,
                    ls_iterations=cfg.sim.mujoco.ls_iterations, njmax=cfg.sim.njmax, nconmax=cfg.sim.nconmax,
                    integrator=cfg.sim.mujoco.integrator, solver=cfg.sim.mujoco.solver,
                    impratio=cfg.sim.mujoco.impratio, cone=cfg.sim.mujoco.cone, tolerance=cfg.sim.mujoco.tolerance,
                    ccd_iterations=cfg.sim.mujoco.ccd_iterations, ls_parallel=cfg.sim.ls_parallel),num_substeps=1)))
        class PhysicsEnvironment(ManagerBasedEnv):
            def load_managers(self):
                pass
        self._launch = launch_simulation(native_cfg,{'headless':True})
        self._launch.__enter__()
        self.native = None
        try:
            self.native=PhysicsEnvironment(native_cfg)
            self.sim=NewtonSimulation(self.native,cfg.sim)
            self.scene=NewtonScene(cfg.scene,self.native,self.sim)
            self.sim.sync_model()
            self._sim_step_counter=0
            self.extras={};self.obs_buf={}
            self._manual_reset_pending=torch.zeros(cfg.scene.num_envs,dtype=torch.bool,device=device)
            self.common_step_counter=0
            self.episode_length_buf=torch.zeros(cfg.scene.num_envs,dtype=torch.long,device=device)
            self.render_mode=render_mode;self._offline_renderer=None
            self.metadata={**ManagerBasedRlEnv.metadata,'physics_backend':'isaac-newton','render_fps':1/self.step_dt}
            self.load_managers()
            self.sim.sync_model()
            self.setup_manager_visualizers()
        except BaseException:
            self.close()
            raise

    def close(self):
        if hasattr(self,'recorder_manager'):self.recorder_manager.close()
        if self.native is not None:
            self.native.close();self.native=None
        if self._launch is not None:
            self._launch.__exit__(None,None,None);self._launch=None
