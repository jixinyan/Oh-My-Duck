"""Name-resolved CPU MuJoCo state writes for the maintained ground-reset sampler."""

from types import SimpleNamespace
import torch
from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES


class CpuResetState:
    def __init__(self, model, data):
        free = model.joint("trunk_base_freejoint").id
        q = int(model.jnt_qposadr[free])
        v = int(model.jnt_dofadr[free])
        self.root_velocity = slice(v, v + 6)
        self.indexing = SimpleNamespace(
            free_joint_q_adr=torch.arange(q, q + 7),
            joint_q_adr=torch.tensor([int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES]),
        )
        self.data = self
        self.qpos = torch.from_numpy(data.qpos).unsqueeze(0)
        self.qvel = torch.from_numpy(data.qvel).unsqueeze(0)

    @property
    def joint_pos(self):
        return self.qpos[:, self.indexing.joint_q_adr]

    def find_joints(self, pattern):
        return list(range(14)), list(JOINT_NAMES)

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.qpos[env_ids[:, None], self.indexing.free_joint_q_adr] = pose.to(self.qpos.dtype)

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.qvel[env_ids, self.root_velocity] = velocity.to(self.qvel.dtype)

    def write_joint_position_to_sim(self, position, env_ids):
        self.qpos[env_ids[:, None], self.indexing.joint_q_adr] = position.to(self.qpos.dtype)


def sample_ground_pose(model, data, parameters):
    from oh_my_duck.rl.mdp.events import set_random_ground_state

    asset = CpuResetState(model, data)
    env = SimpleNamespace(device="cpu", scene={"robot": asset})
    set_random_ground_state(env, torch.tensor([0]), **parameters)
