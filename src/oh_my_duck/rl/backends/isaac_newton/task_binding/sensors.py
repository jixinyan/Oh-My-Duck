"""Task sensors derived from actual Newton contacts and articulation kinematics."""
import re
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from mjlab.sensor import BuiltinSensor, ContactSensor, TerrainHeightSensor
from mjlab.sensor.contact_sensor import ContactData, _AirTimeState
from mjlab.sensor.sensor import Sensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightData
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply, yaw_quat


class NewtonContactSensor(ContactSensor):
    def __init__(self, cfg, robot, simulation):
        super().__init__(cfg)
        self.robot, self.simulation = robot, simulation
        self._data, self._device = simulation.data, simulation.device
        if cfg.num_slots != 1 or cfg.history_length or set(cfg.fields) - {'found','force'}:
            raise ValueError(f'Unsupported contact configuration: {cfg}')
        self.is_feet = cfg.primary.mode == 'geom' and cfg.secondary.mode == 'body' and cfg.secondary.pattern == 'terrain'
        self.is_self = cfg.primary.mode == 'subtree' and cfg.secondary.mode == 'subtree' and cfg.primary.pattern == cfg.secondary.pattern == 'trunk_base'
        if not self.is_feet and not self.is_self:
            raise ValueError('Contact matching must be explicitly bound')
        if self.is_feet:
            local, _ = robot.find_geoms(cfg.primary.pattern)
            self.primary_ids = robot.indexing.geom_ids[local]
            if torch.any(self.primary_ids < 0):
                raise ValueError('Task contact geometry is absent from Newton')
        else:
            self.primary_ids = None
        width = len(self.primary_ids) if self.is_feet else 1
        if cfg.track_air_time:
            zeros = lambda: torch.zeros(simulation.num_envs, width, device=self._device)
            self._air_time_state = _AirTimeState(zeros(),zeros(),zeros(),zeros(),torch.zeros(simulation.num_envs,device=self._device))
        self._force_wp = wp.zeros(simulation.wp_data.naconmax, dtype=wp.spatial_vector, device=self._device)
        self._contact_ids_wp = wp.array(torch.arange(simulation.wp_data.naconmax,device=self._device,dtype=torch.int32),dtype=int,device=self._device)

    def _extract_sensor_data(self):
        sim = self.simulation
        contacts = sim.wp_data.contact
        geom = wp.to_torch(contacts.geom).long()
        world = wp.to_torch(contacts.worldid).long()
        valid = torch.arange(len(world),device=self._device) < wp.to_torch(sim.wp_data.nacon)[0]
        safe_world = world.clamp(0,sim.num_envs-1)
        self._force_wp.zero_()
        mjw.contact_force(sim.wp_model,sim.wp_data,self._contact_ids_wp,True,self._force_wp)
        force = wp.to_torch(self._force_wp)[:,:3]
        body = torch.as_tensor(sim.mj_model.geom_bodyid,device=self._device)
        ground0, ground1 = body[geom[:,0].clamp_min(0)]==0, body[geom[:,1].clamp_min(0)]==0
        found, forces = [], []
        ids = self.primary_ids if self.is_feet else [None]
        for primary in ids:
            if self.is_feet:
                primary0 = (geom[:,0]==primary) & ground1
                primary1 = (geom[:,1]==primary) & ground0
                mask = valid & (primary0 | primary1)
                signed = torch.where(primary0[:,None],-force,force)
            else:
                mask = valid & ~ground0 & ~ground1
                signed = force
            counts = torch.zeros(sim.num_envs,device=self._device)
            counts.scatter_add_(0,safe_world,mask.float())
            net = torch.zeros(sim.num_envs,3,device=self._device)
            net.index_add_(0,safe_world,signed*mask[:,None])
            found.append(counts);forces.append(net)
        return ContactData(found=torch.stack(found,1),force=torch.stack(forces,1) if 'force' in self.cfg.fields else None)


class NewtonBuiltinSensor(BuiltinSensor):
    def __init__(self, name, reference, robot, simulation):
        super().__init__(name=name)
        self.robot, self.simulation = robot, simulation
        spec = reference.sensor(name)
        self.kind = int(spec.type[0])
        obj = int(spec.objid[0])
        if self.kind in (1,2,3):
            self.site = robot.site_names.index(reference.site(obj).name.removeprefix('robot/'))
        elif self.kind in (27,37):
            self.body = robot.body_names.index(reference.body(obj).name.removeprefix('robot/'))
        else:
            raise ValueError(f'Unsupported builtin sensor: {name}, type {self.kind}')

    def _compute_data(self):
        robot, sim = self.robot, self.simulation
        if self.kind == 27:
            return robot.data.body_link_quat_w[:, self.body]
        if self.kind == 37:
            return sim.data.subtree_angmom[:, robot.indexing.body_ids[self.body]]
        pose = robot.data.site_pose_w[:, self.site]
        body = robot.indexing.body_ids[robot.data.site_parent_ids[self.site]]
        root = sim.mj_model.body_rootid[int(body)]
        velocity = sim.data.cvel[:, body]
        offset = pose[:,:3] - sim.data.subtree_com[:, root]
        angular = quat_apply_inverse(pose[:,3:],velocity[:,:3])
        linear = quat_apply_inverse(pose[:,3:],velocity[:,3:] - torch.linalg.cross(offset,velocity[:,:3]))
        if self.kind == 3:
            return angular
        if self.kind == 2:
            return linear
        acceleration = sim.data.cacc[:, body]
        return quat_apply_inverse(pose[:,3:],acceleration[:,3:] - torch.linalg.cross(offset,acceleration[:,:3])) + torch.linalg.cross(angular,linear)


class NewtonFlatHeightSensor(TerrainHeightSensor):
    """Exact vertical ray/plane intersection for the checked flat-ground recipe.

    Unsupported ray directions or extra group-zero geometry fail instead of
    pretending a height estimate is a ray cast on arbitrary terrain.
    """
    def __init__(self,cfg,robot,simulation,reference):
        Sensor.__init__(self)
        self.cfg,self.robot,self.simulation = cfg,robot,simulation
        allowed = [i for i in range(reference.ngeom) if reference.geom_group[i] in cfg.include_geom_groups]
        if cfg.ray_alignment!='yaw' or cfg.reduction!='min' or len(allowed)!=1 or reference.geom_type[allowed[0]]!=mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError('Only the verified flat vertical terrain scan is bound')
        self.site_ids = [robot.site_names.index(frame.name) for frame in cfg.frame]
        self.offsets,self.directions = cfg.pattern.generate_rays(None,simulation.device)
        if not torch.allclose(self.directions,torch.tensor([0.,0.,-1.],device=simulation.device).expand_as(self.directions)) or torch.any(self.offsets[:,2]!=0):
            raise ValueError('Flat height binding requires horizontal offsets and downward rays')
        self.plane_z = float(reference.geom_pos[allowed[0],2])

    def _compute_data(self):
        pose = self.robot.data.site_pose_w[:,self.site_ids]
        b,f,_ = pose.shape;n=len(self.offsets)
        quat = yaw_quat(pose[...,3:]).unsqueeze(2).expand(b,f,n,4)
        origins = pose[...,:3].unsqueeze(2)+quat_apply(quat,self.offsets.expand(b,f,n,3))
        distance = origins[...,2]-self.plane_z
        hit = (distance>=0)&(distance<=self.cfg.max_distance)
        positions=origins.clone();positions[...,2]=torch.where(hit,self.plane_z,origins[...,2])
        normals=torch.zeros_like(origins);normals[...,2]=hit.float()
        distances=torch.where(hit,distance,-1.)
        heights=(pose[...,2]-self.plane_z).clamp(0,self.cfg.max_distance)
        return TerrainHeightData(distances=distances.reshape(b,-1),normals_w=normals.reshape(b,-1,3),hit_pos_w=positions.reshape(b,-1,3),
            pos_w=pose[:,0,:3],quat_w=pose[:,0,3:],frame_pos_w=pose[...,:3],frame_quat_w=pose[...,3:],heights=heights)
