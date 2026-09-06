"""Canonical task state backed by actual Newton solver arrays and Isaac writes.

Task-facing quaternions are WXYZ. Isaac's state API takes XYZW. Geometry and
servo indices are resolved by names rather than relying on either traversal order.
"""

from types import SimpleNamespace
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.entity.data import EntityData, compute_velocity_from_cvel
from mjlab.utils.string import resolve_expr
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch


def unique_suffix(names, suffix):
    ids = [i for i, name in enumerate(names) if name == suffix or name.endswith("_" + suffix)]
    if len(ids) != 1:
        raise ValueError(f"Ambiguous Newton mapping for {suffix}: {[names[i] for i in ids]}")
    return ids[0]


class NewtonEntityData(EntityData):
    @property
    def site_pose_w(self):
        body_pose = self.body_link_pose_w[:, self.site_parent_ids]
        position = body_pose[..., :3] + quat_apply(
            body_pose[..., 3:7], self.site_local_pos.expand(body_pose.shape[0], -1, -1)
        )
        rotation = quat_mul(body_pose[..., 3:7], self.site_local_quat.expand(body_pose.shape[0], -1, -1))
        return torch.cat((position, rotation), dim=-1)

    @property
    def site_vel_w(self):
        ids = self.indexing.body_ids[self.site_parent_ids]
        return compute_velocity_from_cvel(
            self.site_pos_w, self.data.subtree_com[:, self.site_root_ids], self.data.cvel[:, ids]
        )

    @property
    def actuator_force(self):
        return self.qfrc_actuator

    def write_root_pose(self, pose, env_ids=None):
        indices = self.entity.env_ids(env_ids)
        self.entity.native.write_root_link_pose_to_sim_index(
            root_pose=pose[:, [0, 1, 2, 4, 5, 6, 3]], env_ids=indices
        )
        super().write_root_pose(pose, indices)

    def write_root_velocity(self, velocity, env_ids=None):
        indices = self.entity.env_ids(env_ids)
        self.entity.native.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=indices)
        super().write_root_velocity(velocity, indices)

    def write_joint_position(self, position, joint_ids=None, env_ids=None):
        indices = self.entity.env_ids(env_ids)
        joint_ids = slice(None) if joint_ids is None else joint_ids
        self.entity.native.write_joint_position_to_sim_index(
            position=position, joint_ids=self.entity.native_joint_ids[joint_ids], env_ids=indices
        )
        super().write_joint_position(position, joint_ids, indices)

    def write_joint_velocity(self, velocity, joint_ids=None, env_ids=None):
        indices = self.entity.env_ids(env_ids)
        joint_ids = slice(None) if joint_ids is None else joint_ids
        self.entity.native.write_joint_velocity_to_sim_index(
            velocity=velocity, joint_ids=self.entity.native_joint_ids[joint_ids], env_ids=indices
        )
        super().write_joint_velocity(velocity, joint_ids, indices)


class NewtonEntity(Entity):
    def __init__(self, cfg, native, simulation):
        super().__init__(cfg)
        self.native, self.simulation = native, simulation
        self.device = simulation.device
        count, model = native.num_instances, simulation.mj_model
        reference = self.compile()
        self.reference_model = reference
        device = simulation.device
        self._all_env_ids = torch.arange(count, device=device)
        self.native_joint_ids = torch.tensor(
            [native.joint_names.index(name) for name in self.joint_names], device=device
        )
        names = [model.joint(i).name for i in range(model.njnt)]
        joints = [unique_suffix(names, name) for name in self.joint_names]
        body_names = [model.body(i).name for i in range(model.nbody)]
        # USD disambiguates a body that shares a servo name with suffix _1.
        bodies = [
            unique_suffix(body_names, name + ("_1" if name in self.joint_names else ""))
            for name in self.body_names
        ]
        root = bodies[0]
        free_joints = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
        if len(free_joints) != 1:
            raise ValueError("Expected one floating Microduck articulation per world")
        free = free_joints[0]
        tensor = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
        empty = tensor([])
        # Only compiled collision geometry is present in Newton. Selectors refer
        # to named task geoms (the feet); unavailable visual geoms fail on use.
        labels = [model.geom(i).name for i in range(model.ngeom)]
        geom_ids = []
        for name in self.geom_names:
            matches = [i for i, label in enumerate(labels) if name and "/" + name + "/" in label]
            geom_ids.append(matches[0] if len(matches) == 1 else -1)
        self.indexing = SimpleNamespace(
            root_body_id=root,
            body_ids=tensor(bodies),
            joint_ids=tensor(joints),
            geom_ids=tensor(geom_ids),
            site_ids=tensor(range(self.num_sites)),
            ctrl_ids=empty,
            tendon_ids=empty,
            pair_ids=empty,
            joint_q_adr=tensor(model.jnt_qposadr[joints]),
            joint_v_adr=tensor(model.jnt_dofadr[joints]),
            free_joint_q_adr=tensor(range(int(model.jnt_qposadr[free]), int(model.jnt_qposadr[free]) + 7)),
            free_joint_v_adr=tensor(range(int(model.jnt_dofadr[free]), int(model.jnt_dofadr[free]) + 6)),
        )
        limits = simulation.model.jnt_range[:, joints].clone()
        midpoint = limits.mean(-1)
        radius = (limits[..., 1] - limits[..., 0]) * cfg.articulation.soft_joint_pos_limit_factor / 2
        initial_joints = (
            reference.key(0).qpos[7:]
            if cfg.init_state.joint_pos is None
            else resolve_expr(cfg.init_state.joint_pos, self.joint_names, 0.0)
        )
        default_pos = torch.tensor(initial_joints, device=device, dtype=torch.float32).repeat(count, 1)
        default_vel = torch.tensor(
            resolve_expr(cfg.init_state.joint_vel, self.joint_names, 0.0), device=device
        ).repeat(count, 1)
        initial = cfg.init_state
        default_root = torch.tensor(
            (*initial.pos, *initial.rot, *initial.lin_vel, *initial.ang_vel), device=device
        ).repeat(count, 1)
        zeros = lambda width: torch.zeros(count, width, device=device)
        self._data = NewtonEntityData(
            indexing=self.indexing,
            data=simulation.motor_data,
            model=simulation.model,
            device=device,
            default_root_state=default_root,
            default_joint_pos=default_pos,
            default_joint_vel=default_vel,
            default_joint_pos_limits=limits.clone(),
            joint_pos_limits=limits.clone(),
            soft_joint_pos_limits=torch.stack((midpoint - radius, midpoint + radius), -1),
            gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(count, 1),
            forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(count, 1),
            is_fixed_base=False,
            is_articulated=True,
            is_actuated=True,
            joint_pos_target=zeros(self.num_joints),
            joint_vel_target=zeros(self.num_joints),
            joint_effort_target=zeros(self.num_joints),
            tendon_len_target=zeros(0),
            tendon_vel_target=zeros(0),
            tendon_effort_target=zeros(0),
            site_effort_target=zeros(0),
            encoder_bias=zeros(self.num_joints),
        )
        self._data.entity = self
        sites = [reference.site(name).id for name in self.site_names]
        self._data.site_parent_ids = tensor(
            [self.body_names.index(reference.body(int(reference.site_bodyid[i])).name) for i in sites]
        )
        self._data.site_root_ids = tensor(
            model.body_rootid[np.array(bodies)[self._data.site_parent_ids.cpu().numpy()]]
        )
        self._data.site_local_pos = torch.tensor(
            reference.site_pos[sites], device=device, dtype=torch.float32
        ).unsqueeze(0)
        self._data.site_local_quat = torch.tensor(
            reference.site_quat[sites], device=device, dtype=torch.float32
        ).unsqueeze(0)
        actuator = native.actuators["official_bam"]
        if actuator.official is None:
            actuator._initialize_bam()
        # DR and reset use the same fitted motor instance that applies effort in Isaac.
        self._actuators = [actuator.official]

    def find_geoms(self, *args, **kwargs):
        ids, names = super().find_geoms(*args, **kwargs)
        if getattr(self, "indexing", None) is not None and any(
            int(self.indexing.geom_ids[i]) < 0 for i in ids
        ):
            raise ValueError(f"Geometry selector includes shapes absent from Newton: {names}")
        return ids, names

    def env_ids(self, ids):
        return (
            self._all_env_ids if ids is None else (self._all_env_ids[ids] if isinstance(ids, slice) else ids)
        )

    def reset(self, env_ids=None):
        self._data.clear_state(env_ids)
        self.native.reset(self.env_ids(env_ids))

    def write_data_to_sim(self):
        self.native.set_joint_position_target_index(
            target=self.data.joint_pos_target, joint_ids=self.native_joint_ids
        )
        self.native.set_joint_velocity_target_index(
            target=self.data.joint_vel_target, joint_ids=self.native_joint_ids
        )
        self.native.set_joint_effort_target_index(
            target=self.data.joint_effort_target, joint_ids=self.native_joint_ids
        )

    def update(self, dt):
        # Native scene/actuator lifecycle handles the physics substep.
        pass
