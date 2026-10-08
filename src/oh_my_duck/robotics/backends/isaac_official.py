from __future__ import annotations

import base64
from copy import copy
from io import BytesIO
import math
import json
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Callable
from uuid import uuid4

import mujoco
import mujoco_warp as mjw
import numpy as np
from PIL import Image, ImageDraw
import torch
import warp as wp

from oh_my_duck.robotics.backends.simulation import (
    CpuMujocoBamBackend, MotionBusyError, SimulationBackend, _utc_now,
)
from oh_my_duck.robotics.microduck.protocol import HOME, JOINT_NAMES
from oh_my_duck.robotics.policies.catalogue import PolicyCatalogue
from oh_my_duck.robotics.microduck.sim_sensors import camera_optical_pose, tof_directions
from oh_my_duck.perception.rgbd import measure_target
from oh_my_duck.core.paths import project_root
from oh_my_duck.robotics.runtime_startup import RuntimeStartup
from oh_my_duck.infrastructure.usd_runtime import verify_usd_runtime


@wp.kernel
def _clip_camera_rays(rays: wp.array(dtype=wp.vec3f, ndim=4), near: float):
    camera, row, column = wp.tid()
    direction = rays[camera, row, column, 1]
    rays[camera, row, column, 0] = direction * (near / -direction[2])


class IsaacNewtonBamBackend(CpuMujocoBamBackend):
    def __init__(self, *, robot_id: str, catalog_dir: Path, scene_path: Path,
                 device: str, floor_height_m: float = 0.0,
                 scene_id: str = "isaac_external", robot_model: str = "allcollisions",
                 provenance_path: Path | None = None, public_map_path: Path | None = None,
                 observer_renderer: str = "newton_warp", startup: RuntimeStartup | None = None,
                 policy_registry: Path | None = None):
        self._startup = RuntimeStartup(scene_id, robot_model) if startup is None else startup
        verify_usd_runtime()
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg, SimulationCfg, UsdFileCfg
        from isaaclab_tasks.utils import launch_simulation
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        from mjlab.sim import MujocoCfg, SimulationCfg as TaskSimulationCfg
        from oh_my_duck.rl.backends.isaac_newton.bam_actuator import OfficialBamActuatorCfg
        from oh_my_duck.rl.backends.isaac_newton.config import SceneCfg
        from oh_my_duck.rl.backends.isaac_newton.paths import require_asset
        from oh_my_duck.rl.backends.isaac_newton.task_binding.collisions import configure_scene
        from oh_my_duck.rl.backends.isaac_newton.task_binding.environment import PhysicsOnlyTerms
        from oh_my_duck.rl.backends.isaac_newton.task_binding.entity import NewtonEntity
        from oh_my_duck.rl.backends.isaac_newton.task_binding.manager import OfficialTaskSolverCfg
        from oh_my_duck.rl.backends.isaac_newton.task_binding.simulation import NewtonSimulation
        from oh_my_duck.rl.backends.isaac_newton.asset_names import get_isaac_allcollisions_cfg, get_isaac_rollers_cfg

        self._startup.mark("runtime_validation")
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("Isaac/Newton requires an explicitly selected CUDA device")
        factories = {"allcollisions": get_isaac_allcollisions_cfg,
                     "groundcontact_rollers": get_isaac_rollers_cfg}
        if robot_model not in factories or not math.isfinite(floor_height_m):
            raise ValueError("Runtime requires a supported official robot model and finite floor height")
        if observer_renderer not in {"newton_warp", "isaac_rtx"}:
            raise ValueError("Unknown observer renderer")
        self.robot_model, self.observer_renderer = robot_model, observer_renderer
        torch.cuda.set_device(device)
        SimulationBackend.__init__(self, robot_id=robot_id, backend="isaac-newton",
                                   policy_path=None, task_id=scene_id)
        self._np, self._mujoco = np, mujoco
        self.catalog = PolicyCatalogue(catalog_dir, policy_registry)
        self.active_policy = self.catalog.get("roller" if robot_model == "groundcontact_rollers" else "velstand")
        self.policy_sha256 = self.active_policy.sha256
        self._scene_path = scene_path.resolve(strict=True)
        self.scene_id = scene_id
        self.floor_height_m = float(floor_height_m)
        self.provenance = None if provenance_path is None else {
            "path": str(provenance_path), "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest()}
        self.public_map = None if public_map_path is None else json.loads(public_map_path.read_text())
        self._owner_thread = threading.get_ident()
        self._stop_requested = threading.Event()
        self._renderer = self._observer_renderer = None
        # 全部碰撞几何的坐姿使用官方 sitstand 的接触求解参数。
        iterations, ls_iterations = (30, 50) if robot_model == "allcollisions" else (10, 20)
        task_sim_cfg = TaskSimulationCfg(nconmax=200 if robot_model == "allcollisions" else 35, njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=iterations, ls_iterations=ls_iterations))
        scene = SceneCfg(num_envs=1, env_spacing=1.0)
        scene.terrain = None
        scene.environment = AssetBaseCfg(
            prim_path="/World/Environment", spawn=UsdFileCfg(usd_path=str(self._scene_path)))
        scene.robot.spawn.usd_path = str(require_asset(robot_model))
        configure_scene(scene, robot_model)
        scene.robot.actuators = {"official_bam": OfficialBamActuatorCfg(
            joint_names_expr=list(JOINT_NAMES), deployment=True)}
        scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/HeadCamera", width=320, height=240,
            data_types=["rgb", "depth", "instance_segmentation_fast"], update_period=0.0, update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(clipping_range=(0.01, 30.0)),
            renderer_cfg=NewtonWarpRendererCfg(colorize_instance_segmentation=False, enable_shadows=True))
        if observer_renderer == "isaac_rtx":
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            observer_cfg = IsaacRtxRendererCfg()
        else:
            observer_cfg = NewtonWarpRendererCfg(colorize_instance_segmentation=False, enable_shadows=True)
        scene.observer_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ObserverCamera", width=1280, height=720,
            data_types=(["rgb"] if observer_renderer == "isaac_rtx" else ["rgb", "instance_segmentation_fast"]),
            update_period=0.0, update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(clipping_range=(0.01, 100.0)),
            renderer_cfg=observer_cfg)
        native_cfg = ManagerBasedEnvCfg(
            scene=scene, decimation=1, actions=PhysicsOnlyTerms(),
            observations=PhysicsOnlyTerms(), events=PhysicsOnlyTerms(), seed=0,
            sim=SimulationCfg(device=device, dt=0.005, render_interval=4,
                              physics=NewtonCfg(solver_cfg=OfficialTaskSolverCfg(
                                  robot_model=robot_model, iterations=iterations, ls_iterations=ls_iterations,
                                  njmax=5000, nconmax=5000), num_substeps=1,
                                  use_cuda_graph=robot_model != "allcollisions")))
        self._startup.mark("simulation_launch")
        self._launch = launch_simulation(native_cfg, {"headless": True, "device": device})
        self._launch.__enter__()
        self._startup.mark("environment_construction")
        self.native = ManagerBasedEnv(native_cfg)
        self._startup.mark("bam_binding")
        self.sim = NewtonSimulation(self.native, task_sim_cfg)
        self.robot = NewtonEntity(factories[robot_model](),
                                  self.native.scene["robot"], self.sim)
        self._servo_joint_ids = [self.robot.joint_names.index(name) for name in JOINT_NAMES]
        self.model = copy(self.sim.mj_model)
        self._startup.mark("scene_and_sensor_binding")
        self._geometry_normalizations = NewtonManager._builder.omd_scene_transform_audit
        audit_bytes = (json.dumps(self._geometry_normalizations, indent=2, allow_nan=False) + "\n").encode()
        audit_hash = hashlib.sha256(audit_bytes).hexdigest()
        audit_path = project_root() / "outputs/runtime-scene-audits" / f"{audit_hash}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            if audit_path.read_bytes() != audit_bytes:
                raise RuntimeError("Stored scene transform audit differs from its hash")
        else:
            with audit_path.open("xb") as audit_file:
                audit_file.write(audit_bytes)
        self._geometry_audit_reference = {"count": len(self._geometry_normalizations),
                                          "path": str(audit_path), "sha256": audit_hash}
        self.data = mujoco.MjData(self.model)
        self._source_target_ranges = np.tile([-10.0, 10.0], (14, 1))
        reference = self.robot.reference_model
        self._reference_camera_id = reference.camera("head_camera").id
        self._reference_tof_id = reference.site("tof").id
        self._camera_body = self.robot.body_names.index(
            reference.body(int(reference.cam_bodyid[self._reference_camera_id])).name.removeprefix("robot/"))
        self._tof_body = self.robot.body_names.index(
            reference.body(int(reference.site_bodyid[self._reference_tof_id])).name.removeprefix("robot/"))
        self._camera_local_position = reference.cam_pos[self._reference_camera_id].copy()
        self._camera_local_position, self._camera_local_quaternion = camera_optical_pose(reference)
        self._clipped_cameras: set[str] = set()
        self._render_evidence: dict[str, dict] = {}
        self._shape_labels = list(NewtonManager._builder.shape_label)
        self._tof_local_position = reference.site_pos[self._reference_tof_id].copy()
        self._tof_local_quaternion = reference.site_quat[self._reference_tof_id].copy()
        self._robot_geom_ids = set(np.flatnonzero(
            self.model.body_rootid[self.model.geom_bodyid] == self.robot.indexing.root_body_id).tolist())
        self._environment_geom_ids = [i for i in range(self.model.ngeom)
                                      if self.model.geom(i).name.startswith("/World/Environment/")]
        if not self._environment_geom_ids:
            raise RuntimeError("Newton solver contains no external scene geometry")
        # MJCF 的机器人 collision group 为 3；ToF 只读取外部环境 group 0。
        self.model.geom_group[list(self._robot_geom_ids)] = 3
        self.model.geom_group[self._environment_geom_ids] = 0
        self._tof_geom_groups = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        ground_paths = [] if self.public_map is None else self.public_map.get("ground_collider_paths", [])
        self._ground_geom_ids = {i for i in self._environment_geom_ids
                                 if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                                 or any(path in self.model.geom(i).name for path in ground_paths)}
        if not self._ground_geom_ids:
            raise RuntimeError("Scene requires identified native ground collision geometry")
        self._contact_ids_wp = wp.from_torch(torch.arange(
            self.sim.wp_data.naconmax, device=device, dtype=torch.int32))
        self._contact_force_wp = wp.zeros(self.sim.wp_data.naconmax,
                                          dtype=wp.spatial_vector, device=device)
        self.policy_inference = SimpleNamespace(
            command=np.zeros(13, dtype=np.float32), last_action=np.zeros(14, dtype=np.float32),
            get_observations=self._policy_observation)
        self.controller = SimpleNamespace(q_target=np.asarray(HOME, dtype=np.float32).copy())
        self._pending_inference = None
        self._applied_requests = {}
        self._used_action_requests = set()
        self._stopped_samples = 0
        self._selected_at_s = 0.0
        self._goal = None
        self._goal_held_ticks = 0
        self._goal_checked_sequence = -1
        self._requested_command = {"twist": (0.0,) * 3, "head": (0.0,) * 4,
                                   "body": (0.0,) * 6, "posture": "stand"}
        self._reset_contact_evidence()
        self._sensor_rng = np.random.default_rng(0)
        self._startup.mark("runtime_ready", ready=True)

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Newton control must run on its owner thread")

    def _snapshot(self) -> None:
        self.data.qpos[:] = self.sim.data.qpos[0].detach().cpu().numpy()
        self.data.qvel[:] = self.sim.data.qvel[0].detach().cpu().numpy()
        self.data.time = self._simulation_time_s
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Newton GPU state contains nonfinite values")
        # 当前 GPU 状态提供几何射线查询；此处没有执行 CPU 物理步。
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)

    def _policy_observation(self) -> np.ndarray:
        data = self.robot.data
        values = np.concatenate((
            data.root_link_ang_vel_b[0].detach().cpu().numpy(),
            data.projected_gravity_b[0].detach().cpu().numpy(),
            data.joint_pos[0, self._servo_joint_ids].detach().cpu().numpy() - np.asarray(HOME),
            data.joint_vel[0, self._servo_joint_ids].detach().cpu().numpy(), self.policy_inference.last_action,
            self.policy_inference.command)).astype(np.float32)
        if values.shape != (61,) or not np.isfinite(values).all():
            raise FloatingPointError("Invalid Newton 61-dimensional policy observation")
        return values

    def _native_state(self) -> dict:
        data = self.robot.data
        position = data.root_link_pos_w[0].detach().cpu().numpy()
        quaternion = data.root_link_quat_w[0].detach().cpu().numpy()
        linear = data.root_link_lin_vel_b[0].detach().cpu().numpy()
        angular = data.root_link_ang_vel_b[0].detach().cpu().numpy()
        world = data.root_link_lin_vel_w[0].detach().cpu().numpy()
        gravity = data.projected_gravity_b[0].detach().cpu().numpy()
        tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
        yaw = math.atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                         1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
        height = float(position[2] - self.floor_height_m)
        count = int(wp.to_torch(self.sim.wp_data.nacon)[0].item())
        return {"body_position_m": position.tolist(), "body_quaternion_wxyz": quaternion.tolist(),
                "body_twist": [float(linear[0]), float(linear[1]), float(angular[2])],
                "body_twist_world": [float(world[0]), float(world[1]), float(angular[2])],
                "angular_velocity_rad_s": angular.tolist(), "projected_gravity": gravity.tolist(),
                "joint_names": list(JOINT_NAMES),
                "joint_position_rad": data.joint_pos[0, self._servo_joint_ids].detach().cpu().tolist(),
                "joint_velocity_rad_s": data.joint_vel[0, self._servo_joint_ids].detach().cpu().tolist(),
                "passive_joints": {name: {"position_rad": float(data.joint_pos[0, i]),
                                          "velocity_rad_s": float(data.joint_vel[0, i])}
                                   for i, name in enumerate(self.robot.joint_names) if name.startswith("passive_")},
                "height_m": height, "tilt_rad": tilt, "fallen": height < 0.06 or tilt > 1.3,
                "odometry": {"x_m": float(position[0]), "y_m": float(position[1]),
                             "yaw_rad": yaw, "frame_id": "world", "source": "Newton GPU state"},
                "native_contact_count": count,
                "contact_evidence": {"non_ground_external_contact_samples_total": self._non_ground_contact_samples_total,
                    "non_ground_external_contact_control_steps": self._non_ground_contact_control_steps,
                    "first_non_ground_external_contact_sequence": self._first_non_ground_contact_sequence,
                    "current_control_non_ground_external": self._current_control_non_ground_contacts,
                    "current_control_ground_contact_samples": self._current_control_ground_contact_samples,
                    "current_control_self_contact_samples": self._current_control_self_contact_samples}}

    def _frame_pose(self, body_index, local_position, local_quaternion):
        pose = self.robot.data.body_link_pose_w[0, body_index].detach().cpu().numpy()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, pose[3:7])
        position = pose[:3] + rotation.reshape(3, 3) @ local_position
        quaternion = np.zeros(4)
        mujoco.mju_mulQuat(quaternion, pose[3:7], local_quaternion)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return position, matrix.reshape(3, 3)

    def _render_rgb(self, name: str, eye, lookat, up, width: int, height: int):
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch
        camera = self.native.scene[name]
        forward = np.asarray(lookat) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up))
        right /= np.linalg.norm(right)
        corrected_up = np.cross(right, forward)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, np.column_stack((right, corrected_up, -forward)).ravel())
        camera.set_world_poses(
            positions=torch.as_tensor(np.asarray(eye)[None], device=self.sim.device, dtype=torch.float32),
            orientations=torch.as_tensor(quaternion[[1, 2, 3, 0]][None], device=self.sim.device, dtype=torch.float32),
            convention="opengl")
        self.native.sim.render()
        if camera.cfg.renderer_cfg.renderer_type == "newton_warp" and name not in self._clipped_cameras:
            rays = camera._render_data.camera_rays
            directions = wp.to_torch(rays)[..., 1, :]
            if not torch.isfinite(directions).all() or not torch.all(directions[..., 2] < 0):
                raise ValueError("Newton camera rays have invalid optical directions")
            wp.launch(_clip_camera_rays, dim=rays.shape[:3],
                      inputs=[rays, camera.cfg.spawn.clipping_range[0]], device=self.sim.device)
            self._clipped_cameras.add(name)
        camera.update(0.0, force_recompute=True)
        rgb = as_torch(camera.data.output["rgb"])[0, ..., :3].detach().cpu().numpy()
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or np.ptp(rgb) == 0:
            raise ValueError("Newton renderer returned invalid RGB")
        if "instance_segmentation_fast" in camera.data.output:
            segmentation = as_torch(camera.data.output["instance_segmentation_fast"])[0, ..., 0].detach().cpu().numpy()
            ids, counts = np.unique(segmentation, return_counts=True)
        else:
            ids, counts = [], []
        visible = [{"shape_id": int(index), "shape": self._shape_labels[int(index)], "pixels": int(count)}
                   for index, count in zip(ids, counts, strict=True) if 0 <= index < len(self._shape_labels)]
        self._render_evidence[name] = {"eye_world_m": np.asarray(eye).tolist(),
                                      "forward_world": forward.tolist(), "up_world": corrected_up.tolist(),
                                      "near_plane_m": camera.cfg.spawn.clipping_range[0],
                                      "rgb_mean": float(rgb.mean()), "rgb_std": float(rgb.std()),
                                      "renderer": camera.cfg.renderer_cfg.renderer_type,
                                      "visible_shapes": visible}
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _sensor_frame(self) -> dict:
        self._snapshot()
        eye, rotation = self._frame_pose(self._camera_body, self._camera_local_position,
                                         self._camera_local_quaternion)
        rgb = self._render_rgb("head_camera", eye, eye - rotation[:, 2], rotation[:, 1], 320, 240)
        origin, tof_rotation = self._frame_pose(self._tof_body, self._tof_local_position,
                                               self._tof_local_quaternion)
        distances, statuses, hit_geoms, raw_hit_geoms, hit_distances = [], [], [], [], []
        geom = np.zeros(1, dtype=np.int32)
        directions = (tof_rotation @ tof_directions().T).T
        for direction in directions:
            raw_hit = mujoco.mj_ray(self.model, self.data, origin, np.ascontiguousarray(direction),
                                    None, 1, -1, geom)
            raw_hit_geoms.append(self.model.geom(int(geom[0])).name if raw_hit >= 0 else None)
            hit = mujoco.mj_ray(self.model, self.data, origin, np.ascontiguousarray(direction),
                                self._tof_geom_groups, 1, -1, geom)
            hit_geoms.append(self.model.geom(int(geom[0])).name if hit >= 0 else None)
            hit_distances.append(float(hit) if hit >= 0 and math.isfinite(hit) else None)
            valid = math.isfinite(hit) and 0 <= hit <= 4.0
            measured = max(0.0, hit + self._sensor_rng.normal(0.0, 0.003 + 0.02 * hit / 4.0)) if valid else 0.0
            distances.append(min(4000, int(measured * 1000.0)))
            statuses.append(5 if valid else 255)
        return {"rgb_png_base64": rgb, "rgb_width": 320, "rgb_height": 240,
                "tof_distance_mm": distances, "tof_status": statuses, "tof_rows": 8, "tof_cols": 8,
                "tof_hit_geoms": hit_geoms, "tof_unfiltered_hit_geoms": raw_hit_geoms,
                "tof_hit_distance_m": hit_distances,
                "tof_ray_origin_world_m": origin.tolist(), "tof_ray_directions_world": directions.tolist(),
                "tof_hit_is_environment": [name is not None and name.startswith("/World/Environment/") for name in hit_geoms],
                "tof_geometry_filter": "Environment group 0; robot MJCF collision group 3 excluded",
                "camera_frame_id": "head_camera", "tof_frame_id": "tof",
                "capture_time_s": self._simulation_time_s,
                "camera_source": "Isaac NewtonWarpRenderer",
                "camera_geometry": self._render_evidence["head_camera"],
                "tof_source": "Newton SolverMuJoCo geometry and current GPU pose"}

    def _reset_contact_evidence(self):
        self._non_ground_contact_samples_total = 0
        self._non_ground_contact_control_steps = 0
        self._first_non_ground_contact_sequence = None
        self._current_control_non_ground_contacts = []
        self._current_control_ground_contact_samples = 0
        self._current_control_self_contact_samples = 0

    def _sample_native_contacts(self, evidence):
        self._contact_force_wp.zero_()
        mjw.contact_force(self.sim.wp_model, self.sim.wp_data,
                          self._contact_ids_wp, False, self._contact_force_wp)
        count = int(wp.to_torch(self.sim.wp_data.nacon)[0].item())
        geom = wp.to_torch(self.sim.wp_data.contact.geom)[:count].cpu().numpy()
        forces = wp.to_torch(self._contact_force_wp)[:count, :3].cpu().numpy()
        ground = self_contact = 0
        for pair, force in zip(geom, forces):
            first, second = int(pair[0]) in self._robot_geom_ids, int(pair[1]) in self._robot_geom_ids
            magnitude = abs(float(force[0]))
            if not math.isfinite(magnitude):
                raise FloatingPointError("Newton contact force contains nonfinite values")
            if not magnitude or not first and not second:
                continue
            if first and second:
                self_contact += 1
            elif int(pair[1] if first else pair[0]) in self._ground_geom_ids:
                ground += 1
            else:
                other = int(pair[1] if first else pair[0])
                name = self.model.geom(other).name
                item = evidence.setdefault(name, {"geom": name, "contact_samples": 0, "max_normal_force_n": 0.0})
                item["contact_samples"] += 1
                item["max_normal_force_n"] = max(item["max_normal_force_n"], magnitude)
                self._non_ground_contact_samples_total += 1
        return ground, self_contact

    def reset_episode(self, seed: int, goal: dict, spawn_pose: dict | None = None) -> dict:
        self._require_owner()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        spawn = {"x_m": 0.0, "y_m": 0.0, "z_m": self.floor_height_m + 0.125,
                 "yaw_rad": 0.0} if spawn_pose is None else spawn_pose
        if set(spawn) != {"x_m", "y_m", "z_m", "yaw_rad"} or not all(math.isfinite(float(v)) for v in spawn.values()):
            raise ValueError("spawn_pose requires finite x_m, y_m, z_m and yaw_rad")
        self.floor_height_m = float(spawn["z_m"]) - 0.125
        self.sim.reset()
        self.native.scene.reset()
        yaw = float(spawn["yaw_rad"])
        pose = torch.tensor([[float(spawn["x_m"]), float(spawn["y_m"]),
                              float(spawn["z_m"]), math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]],
                            device=self.sim.device, dtype=torch.float32)
        self.robot.data.write_root_pose(pose)
        self.robot.data.write_root_velocity(torch.zeros(1, 6, device=self.sim.device))
        positions = self.robot.data.default_joint_pos.clone()
        positions[:, self._servo_joint_ids] = torch.tensor(HOME, device=self.sim.device)
        self.robot.data.write_joint_position(positions)
        self.robot.data.write_joint_velocity(torch.zeros_like(positions))
        self.robot.data.joint_pos_target[:] = positions
        self.robot.write_data_to_sim()
        self.sim.forward()
        self.episode_id = uuid4().hex
        self._sequence = 0
        self._simulation_time_s = 0.0
        self._selected_at_s = 0.0
        self.policy_inference.last_action.fill(0)
        self.controller.q_target[:] = HOME
        self._requested_command = {"twist": (0.0,) * 3, "head": (0.0,) * 4,
                                   "body": (0.0,) * 6, "posture": "stand"}
        self._stop_requested.clear()
        self._pending_inference = None
        self._applied_requests.clear()
        self._used_action_requests.clear()
        self._stopped_samples = 0
        self._reset_contact_evidence()
        self._sensor_rng = np.random.default_rng(seed)
        self.bind_goal(goal)
        return self.observe_control()

    def perception_frame(self) -> dict:
        self._require_owner()
        observed = self.observe_control()
        camera = self.native.scene["head_camera"]
        render = camera._render_data
        depth = wp.to_torch(render.outputs.depth_image)[0, 0].detach().cpu().numpy()
        rays = wp.to_torch(render.camera_rays)[0].detach().cpu().numpy()
        eye, rotation = self._frame_pose(self._camera_body, self._camera_local_position,
                                         self._camera_local_quaternion)
        points = (rays[..., 0, :] + rays[..., 1, :] * depth[..., None]) @ rotation.T + eye
        points[~np.isfinite(depth) | (depth < 0)] = np.nan
        output = BytesIO()
        np.save(output, points.astype(np.float32), allow_pickle=False)
        measured = observed["measurements"]
        return {"episode_id": observed["episode_id"], "sequence": observed["sequence"],
                "observed_at": observed["observed_at"], "rgb_png_base64": measured["rgb_png_base64"],
                "points_world_npy_base64": base64.b64encode(output.getvalue()).decode(),
                "camera_position_m": eye.tolist(), "body_position_m": measured["body_position_m"],
                "yaw_rad": measured["odometry"]["yaw_rad"], "distance_source": "simulator_ground_truth",
                "depth_semantics": "Newton ray-hit meters from actual clipped ray origin",
                "camera_frame_id": measured["camera_frame_id"]}

    def inspect_scene(self, prompt: str) -> dict:
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch
        frame = self.perception_frame()
        camera = self.native.scene["head_camera"]
        segmentation = as_torch(camera.data.output["instance_segmentation_fast"])[0, ..., 0].detach().cpu().numpy()
        points = np.load(BytesIO(base64.b64decode(frame["points_world_npy_base64"])), allow_pickle=False)
        groups = {}
        names = {"desk": ("TableWork", "StandTableWork"), "plant": ("Plant",),
                 "bin": ("TrashCan",), "reception": ("ReceptionTable",),
                 "chair": ("Chair", "Seat"), "door": ("Door",)}
        for index in np.unique(segmentation):
            if not 0 <= index < len(self._shape_labels):
                continue
            shape = self._shape_labels[int(index)]
            if not shape.startswith("/World/Environment/"):
                continue
            asset = "/".join(shape.split("/")[:4])
            labels = [label for label, tokens in names.items() if any(token in asset for token in tokens)]
            if not labels or prompt not in ("objects", *labels):
                continue
            if asset not in groups:
                groups[asset] = {"label": labels[0], "ids": []}
            groups[asset]["ids"].append(int(index))
        image = Image.open(BytesIO(base64.b64decode(frame["rgb_png_base64"]))).convert("RGB")
        draw = ImageDraw.Draw(image)
        targets = []
        for asset, item in groups.items():
            mask = np.isin(segmentation, item["ids"])
            rows, columns = np.nonzero(mask)
            if len(rows) < 8:
                continue
            box = [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]
            target = {"target_id": asset, "label": item["label"], "bbox_xyxy": box,
                      "visible_pixels": len(rows), "detection_source": "simulator_ground_truth",
                      "mask_source": "simulator_ground_truth", "distance_source": "simulator_ground_truth",
                      **measure_target(mask, points, frame["camera_position_m"], frame["body_position_m"], frame["yaw_rad"])}
            targets.append(target)
            draw.rectangle(box, outline="lime", width=2)
            draw.text((box[0], box[1]), item["label"], fill="white")
        output = BytesIO()
        image.save(output, format="PNG")
        return {"episode_id": frame["episode_id"], "sequence": frame["sequence"],
                "observed_at": frame["observed_at"], "prompt": prompt,
                "detection_source": "simulator_ground_truth", "distance_source": "simulator_ground_truth",
                "targets": sorted(targets, key=lambda target: -target["visible_pixels"])[:16],
                "rgb_png_base64": base64.b64encode(output.getvalue()).decode()}

    def bind_goal(self, goal: dict) -> dict:
        self._require_owner()
        if (set(goal) - {"waypoints"} != {"kind", "target_xy_m", "distance_m", "hold_ticks"}
                or goal["kind"] != "point" or len(goal["target_xy_m"]) != 2
                or not all(math.isfinite(float(v)) for v in goal["target_xy_m"])
                or not math.isfinite(float(goal["distance_m"])) or goal["distance_m"] <= 0
                or type(goal["hold_ticks"]) is not int or not 1 <= goal["hold_ticks"] <= 500):
            raise ValueError("Point goal requires target_xy_m, positive distance_m and bounded hold_ticks")
        for waypoint in goal.get("waypoints", []):
            if (set(waypoint) != {"target_xy_m", "distance_m"} or len(waypoint["target_xy_m"]) != 2
                    or not np.isfinite(waypoint["target_xy_m"]).all()
                    or not math.isfinite(waypoint["distance_m"]) or waypoint["distance_m"] <= 0):
                raise ValueError("Route waypoints require finite XY meters and positive radius")
        self._goal = goal.copy()
        self._route_visits = []
        self._route_last_sequence = -1
        self._goal_held_ticks = 0
        self._goal_checked_sequence = -1
        return {"episode_id": self.episode_id, "goal": self._goal,
                "supported_check_ids": ["goal_reached"]}

    def _goal_measurement(self):
        state = self._native_state()
        position = np.asarray(state["body_position_m"])
        distance = float(np.linalg.norm(position[:2] - self._goal["target_xy_m"]))
        reached = distance <= self._goal["distance_m"] and state["height_m"] >= 0.09 and state["tilt_rad"] <= math.radians(25)
        waypoints = self._goal.get("waypoints", [])
        if self._sequence != self._route_last_sequence and len(self._route_visits) < len(waypoints):
            waypoint = waypoints[len(self._route_visits)]
            if (np.linalg.norm(position[:2] - waypoint["target_xy_m"]) <= waypoint["distance_m"]
                    and state["height_m"] >= 0.09 and state["tilt_rad"] <= math.radians(25)):
                self._route_visits.append({"sequence": self._sequence, "position_xy_m": position[:2].tolist()})
            self._route_last_sequence = self._sequence
        reached = reached and len(self._route_visits) == len(waypoints)
        target = {"target_xy_m": self._goal["target_xy_m"], "distance_xy_m": distance,
                  "threshold_m": self._goal["distance_m"], "source": "Newton GPU root pose",
                  "route_visits": self._route_visits.copy(), "required_waypoints": len(waypoints)}
        return reached, target, state, position.tolist()

    def action_spec(self):
        specification = super().action_spec()
        specification["robot_model"] = f"robot_{self.robot_model}"
        specification["head_body_behavior_status"] = "pending_scene_validation"
        return specification

    def public_scene_info(self):
        self._require_owner()
        public_map = None if self.public_map is None else dict(self.public_map)
        if public_map is not None and "obstacles" in public_map:
            position = self._native_state()["body_position_m"][:2]
            centers = [position, self._goal["target_xy_m"]] if self._goal is not None else [position]
            public_map["obstacles"] = [item for item in public_map["obstacles"] if any(
                sum(max(item["min"][i] - center[i], center[i] - item["max"][i], 0) ** 2
                    for i in range(2)) <= 16 for center in centers)]
            public_map["query_regions"] = {"centers_xy_m": centers, "radius_m": 4,
                "returned_obstacle_count": len(public_map["obstacles"]),
                "stored_obstacle_count": len(self.public_map["obstacles"]),
                "coverage": "Intersection with stored authored map; query again after movement"}
        return {"scene_id": self.scene_id, "source": str(self._scene_path), "frame_id": "world",
                "source_kind": "USD imported into Newton SolverMuJoCo",
                "floor_height_m": self.floor_height_m,
                "environment_collision_geom_count": len(self._environment_geom_ids),
                "ground_collision_geoms": [self.model.geom(i).name for i in sorted(self._ground_geom_ids)],
                "signed_scale_normalizations": self._geometry_audit_reference,
                "provenance": self.provenance, "public_map": public_map,
                "goal": self._goal,
                "solver": "Newton SolverMuJoCo", "physics_dt_s": 0.005,
                "solver_iterations": int(self.sim.mj_model.opt.iterations),
                "solver_ls_iterations": int(self.sim.mj_model.opt.ls_iterations),
                "physics_cuda_graph": self.robot_model != "allcollisions",
                "control_hz": 50, "robot_model": self.robot_model,
                "observer_renderer": self.observer_renderer,
                "runtime_startup": self._startup.reference()}

    def list_policies(self):
        result = super().list_policies()
        result["scene"] = self.scene_id
        result["robot_model"] = self.robot_model
        for policy in result["policies"]:
            policy["executable_here"] = (policy["required_robot_mode"] == self.robot_model
                if policy.get("source") == "registered_schema2_package" else
                policy["required_robot_mode"] != "groundcontact_rollers" or self.robot_model == "groundcontact_rollers")
            policy["current_scene_behavior_status"] = "requires_actual_scene_validation"
        return result

    def apply_policy_action(self, action: list[float], request_id: str,
                            expected_sequence: int | None = None,
                            should_stop: Callable[[], bool] | None = None):
        self._require_owner()
        if not request_id:
            raise ValueError("request_id is required")
        existing = self._applied_requests.get(request_id)
        if existing is not None:
            if list(action) != existing["executed_actions"] or expected_sequence not in (None, existing["action_sequence"]):
                raise ValueError("request_id was reused with another action")
            return existing
        if request_id in self._used_action_requests:
            raise ValueError("Interrupted action request cannot be replayed")
        values = np.asarray(action, dtype=np.float32)
        ticket = self._pending_inference
        if (values.shape != (14,) or not np.isfinite(values).all() or ticket is None
                or ticket["episode_id"] != self.episode_id or ticket["sequence"] != self._sequence
                or ticket["policy"] != self.active_policy.name or not np.array_equal(values, ticket["action"])
                or expected_sequence not in (None, self._sequence)):
            raise ValueError("Action differs from the current policy inference ticket")
        if self._stop_requested.is_set() or should_stop is not None and should_stop():
            self._pending_inference = None
            self._used_action_requests.add(request_id)
            raise MotionBusyError("Motion stop was requested before actuation")
        targets = np.asarray(HOME, dtype=np.float32) + values * self.active_policy.action_scale
        if np.any(targets < -10) or np.any(targets > 10):
            raise ValueError("Official policy exceeds the source target range")
        source_sequence = self._sequence
        self.controller.q_target[:] = targets
        self.robot.data.joint_pos_target[:, self._servo_joint_ids] = torch.as_tensor(targets, device=self.sim.device)
        evidence, ground, self_contacts, executed = {}, 0, 0, 0
        with torch.inference_mode():
            for _ in range(4):
                if self._stop_requested.is_set() or should_stop is not None and should_stop():
                    break
                self.robot.write_data_to_sim()
                self.sim.step()
                executed += 1
                self._simulation_time_s += 0.005
                g, s = self._sample_native_contacts(evidence)
                ground += g
                self_contacts += s
        interrupted = executed != 4
        if executed:
            self.policy_inference.last_action[:] = values
            self._sequence += 1
            self._sampled_at = _utc_now()
            self._tick_finished_at = time.monotonic()
            self.data.time = self._simulation_time_s
            self._current_control_non_ground_contacts = list(evidence.values())
            self._current_control_ground_contact_samples = ground
            self._current_control_self_contact_samples = self_contacts
            if evidence:
                self._non_ground_contact_control_steps += 1
                if self._first_non_ground_contact_sequence is None:
                    self._first_non_ground_contact_sequence = self._sequence
            if executed == 4:
                self._update_control_evidence()
            else:
                self._stopped_samples = self._goal_held_ticks = 0
        self._pending_inference = None
        self._used_action_requests.add(request_id)
        result = self.observe_control()
        result.update({"request_id": request_id, "executed_actions": values.tolist() if executed else [],
                       "raw_sim_steps": executed, "interrupted": interrupted,
                       "action_sequence": source_sequence, "policy_name": self.active_policy.name,
                       "policy_sha256": self.active_policy.sha256})
        if not interrupted:
            self._applied_requests[request_id] = result
        return result

    def capture_observer(self, *, include_segmentation: bool = False):
        self._require_owner()
        if include_segmentation:
            raise ValueError("Newton observer segmentation must be configured explicitly")
        lookat = np.asarray(self._native_state()["body_position_m"])
        eye = lookat + np.array([2.0, -2.0, 1.4])
        target = lookat + np.array([0.0, 0.0, 0.35])
        encoded = self._render_rgb("observer_camera", eye, target, (0, 0, 1), 1280, 720)
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "simulation_time_s": self._simulation_time_s, "capture_time_s": self._simulation_time_s,
                "camera_frame_id": "observer_follow", "rgb_width": 1280, "rgb_height": 720,
                "rgb_png_base64": encoded,
                "camera": {"mode": "Newton scene follow camera", "lookat_world_m": target.tolist(),
                           "geometry": self._render_evidence["observer_camera"]}}

    def close(self):
        self.native.close()
        self._launch.__exit__(None, None, None)
