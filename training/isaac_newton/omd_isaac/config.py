"""Explicit Newton/MJWarp diagnostic scene with canonical policy joint ordering."""
from pathlib import Path
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg, reset_joints_by_offset, reset_root_state_uniform, time_out
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, ArticulationRootPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from .contracts import JOINT_NAMES, HOME
from .paths import usd_path
from . import mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd",
        usd_path=str(Path(__file__).parent / "resources/ground_plane.usda"))
    robot = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(usd_path=str(usd_path()), articulation_props=ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.125), joint_pos=dict(zip(JOINT_NAMES, HOME)), joint_vel={".*": 0}),
        actuators={"diagnostic_pd": IdealPDActuatorCfg(joint_names_expr=list(JOINT_NAMES),
            stiffness=5.0, damping=0.1, effort_limit=0.96)})


@configclass
class ActionsCfg:
    joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=list(JOINT_NAMES),
        preserve_order=True, scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=mdp.observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    policy = PolicyCfg()


@configclass
class RewardsCfg:
    upright = RewardTermCfg(func=mdp.upright, weight=1.0)
    posture = RewardTermCfg(func=mdp.pose_error, weight=-0.1)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    fallen = TerminationTermCfg(func=mdp.fallen)


@configclass
class EventsCfg:
    root = EventTermCfg(func=reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}})
    joints = EventTermCfg(func=reset_joints_by_offset, mode="reset", params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class DiagnosticEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=16, env_spacing=1.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    commands = None
    curriculum = None
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim = SimulationCfg(dt=0.005, render_interval=4,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=10, ls_iterations=20), num_substeps=1))
