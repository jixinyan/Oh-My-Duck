"""Official task managers on Isaac Lab's native Newton physics lifecycle.

Managers retain their original order, delays, noise, reward functions, curriculum
and reset behavior. The constructor intentionally replaces only physics and scene
creation from mjlab; no MuJoCo simulation environment is instantiated here.
"""
import torch
from isaaclab.utils.configclass import configclass


@configclass
class PhysicsOnlyTerms:
    """A truthy, empty native manager configuration (an empty dict skips setup)."""
    pass

from mjlab.envs import ManagerBasedRlEnv
from oh_my_duck.rl.backends.isaac_newton.task_binding.simulation import NewtonSimulation
from oh_my_duck.rl.backends.isaac_newton.task_binding.scene import NewtonScene


class NewtonTaskEnvironment(ManagerBasedRlEnv):
    def __init__(self,cfg,*,task,device,render_mode=None):
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from isaaclab_tasks.utils import launch_simulation
        from isaaclab.sim import SimulationCfg
        from isaaclab_newton.physics import NewtonCfg
        from .manager import OfficialTaskSolverCfg
        from oh_my_duck.rl.backends.isaac_newton.config import SceneCfg
        from oh_my_duck.rl.backends.isaac_newton.bam_actuator import OfficialBamActuatorCfg
        from oh_my_duck.rl.backends.isaac_newton.task_binding.collisions import configure_scene
        from oh_my_duck.rl.backends.isaac_newton.paths import require_asset
        from oh_my_duck.rl.backends.isaac_newton.contracts import JOINT_NAMES
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Only headless rgb_array rendering is supported')
        if set(cfg.scene.entities) != {'robot'} or cfg.scene.terrain is None or cfg.scene.terrain.terrain_type != 'plane':
            raise ValueError('Only the explicit flat robot scene is currently bound')
        self.cfg=cfg
        if cfg.seed is not None:self.cfg.seed=self.seed(cfg.seed)
        scene_cfg=SceneCfg(num_envs=cfg.scene.num_envs,env_spacing=cfg.scene.env_spacing)
        scene_cfg.robot.spawn.usd_path=str(require_asset(task.model))
        configure_scene(scene_cfg,task.model)
        if render_mode == 'rgb_array':
            from isaaclab.sensors import CameraCfg
            from isaaclab.sim import PinholeCameraCfg
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            scene_cfg.camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Camera', width=cfg.viewer.width,
                height=cfg.viewer.height, data_types=['rgb'], update_period=0., update_latest_camera_pose=True,
                spawn=PinholeCameraCfg(clipping_range=(0.01, 10.)), renderer_cfg=NewtonWarpRendererCfg())
        scene_cfg.robot.actuators={'official_bam':OfficialBamActuatorCfg(joint_names_expr=list(JOINT_NAMES))}
        native_cfg=ManagerBasedEnvCfg(scene=scene_cfg,decimation=1,actions=PhysicsOnlyTerms(),observations=PhysicsOnlyTerms(),events=PhysicsOnlyTerms(),seed=cfg.seed,
            sim=SimulationCfg(device=device,dt=cfg.sim.mujoco.timestep,render_interval=cfg.decimation,
                physics=NewtonCfg(solver_cfg=OfficialTaskSolverCfg(robot_model=task.model, iterations=cfg.sim.mujoco.iterations,
                    ls_iterations=cfg.sim.mujoco.ls_iterations, njmax=cfg.sim.njmax, nconmax=cfg.sim.nconmax,
                    integrator=cfg.sim.mujoco.integrator, solver=cfg.sim.mujoco.solver,
                    impratio=cfg.sim.mujoco.impratio, cone=cfg.sim.mujoco.cone, tolerance=cfg.sim.mujoco.tolerance,
                    ccd_iterations=cfg.sim.mujoco.ccd_iterations, ls_parallel=cfg.sim.ls_parallel),num_substeps=1)))
        PhysicsEnvironment = ManagerBasedEnv
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
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Cleanup also failed: {cleanup_error!r}")
            raise

    def render(self):
        if self.render_mode is None:
            return None
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch
        camera = self.native.scene['camera']
        lookat = self.scene['robot'].data.root_link_pos_w.clone()
        eye = lookat + torch.tensor([0.6, 0.6, 0.35], device=self.device)
        camera.set_world_poses_from_view(eye, lookat)
        self.native.sim.render()
        camera.update(0., force_recompute=True)
        torch.testing.assert_close(as_torch(camera.data.pos_w), eye, atol=1e-5, rtol=0)
        return as_torch(camera.data.output['rgb'])[0, ..., :3].cpu().numpy()

    def close(self):
        if hasattr(self,'recorder_manager'):self.recorder_manager.close()
        if self.native is not None:
            self.native.close();self.native=None
        if self._launch is not None:
            self._launch.__exit__(None,None,None);self._launch=None
