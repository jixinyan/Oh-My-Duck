"""Ground-state sampling must respect backend joint mappings and subset resets."""
from types import SimpleNamespace
import torch
from oh_my_duck.rl.mdp.events import set_random_ground_state
from oh_my_duck.rl.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg


class RobotData:
    def __init__(self,raw,indexing):self.data,self.indexing=raw,indexing
    @property
    def joint_pos(self):return self.data.qpos[:,self.indexing.joint_q_adr]


class Robot:
    def __init__(self,raw,permutation):
        self.indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7),joint_q_adr=7+permutation)
        self.data=RobotData(raw,self.indexing)
    def find_joints(self,pattern):return list(range(14)),[]
    def write_root_link_pose_to_sim(self,pose,env_ids):self.data.data.qpos[env_ids[:,None],self.indexing.free_joint_q_adr]=pose
    def write_root_link_velocity_to_sim(self,velocity,env_ids):self.data.data.qvel[env_ids,:6]=velocity
    def write_joint_position_to_sim(self,position,env_ids):self.data.data.qpos[env_ids[:,None],self.indexing.joint_q_adr]=position


def fixture(permutation):
    raw=SimpleNamespace(qpos=torch.ones(32,21),qvel=torch.ones(32,20))
    robot=Robot(raw,permutation)
    return SimpleNamespace(device='cpu',sim=SimpleNamespace(data=raw),scene={'robot':robot})


def run_event(event,permutation):
    env=fixture(permutation)
    params=make_microduck_standup_env_cfg().events['set_ground_state'].params.copy()
    params.update(face_down_prob=.25,face_up_prob=.25,sitting_prob=.25,standing_prob=.25)
    torch.manual_seed(99)
    event(env,torch.arange(0,32,2),**params)
    return env


def test_ground_reset_is_independent_of_solver_joint_order():
    canonical=run_event(set_random_ground_state,torch.arange(14))
    permuted=run_event(set_random_ground_state,torch.randperm(14))
    torch.testing.assert_close(canonical.scene['robot'].data.joint_pos,permuted.scene['robot'].data.joint_pos,atol=0,rtol=0)
    torch.testing.assert_close(canonical.sim.data.qpos[:,:7],permuted.sim.data.qpos[:,:7],atol=0,rtol=0)
    assert torch.all(permuted.sim.data.qpos[1::2]==1)
    assert torch.all(permuted.sim.data.qvel[::2,:6]==0)
    assert torch.all(permuted.sim.data.qvel[::2,6:]==1)


def test_reset_fixture_uses_native_entity_interfaces():
    from mjlab.entity import Entity
    import inspect
    for name in ('write_root_link_pose_to_sim', 'write_root_link_velocity_to_sim', 'write_joint_position_to_sim'):
        parameters = inspect.signature(getattr(Entity, name)).parameters
        assert 'env_ids' in parameters
        assert hasattr(Robot, name)
