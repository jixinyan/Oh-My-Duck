from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import math
from pathlib import Path
import threading
import time
from typing import Callable
from uuid import uuid4

from oh_my_duck.core.contracts.identity import ExecutionDomain, Identity
from oh_my_duck.core.contracts.sensors import SensorFrame, Validity
from oh_my_duck.robotics.backends.base import RobotCapabilities, RobotState, StopResult
from oh_my_duck.robotics.microduck.protocol import HOME, JOINT_NAMES


CONTROL_DT = 0.02
STOP_LINEAR_M_S = 0.025
STOP_ANGULAR_RAD_S = 0.08
STOP_SAMPLES = 5
STOP_TIMEOUT_S = 3.0


class MotionBusyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_velocity(vx_m_s: float, vy_m_s: float, yaw_rad_s: float) -> tuple[float, float, float]:
    values = (vx_m_s, vy_m_s, yaw_rad_s)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Velocity components must be finite numbers")
    if abs(vx_m_s) > 0.5 or abs(vy_m_s) > 0.5 or abs(yaw_rad_s) > 2.0:
        raise ValueError("Velocity exceeds this runtime's command limits")
    return tuple(float(value) for value in values)


def validate_policy(path: Path):
    import numpy as np
    from oh_my_duck.rl.artifacts.inference import cpu_session

    if not path.is_file():
        raise FileNotFoundError(path)
    session = cpu_session(str(path))
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Policy must have one observation input and one action output")
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise ValueError("Policy requires a fixed 61-observation/14-action shape")
    metadata = session.get_modelmeta().custom_metadata_map
    if tuple(metadata.get("joint_names", "").split(",")) != JOINT_NAMES:
        raise ValueError("Policy joint order differs from the official order")
    scales = np.asarray([float(value) for value in metadata.get("action_scale", "nan").split(",")])
    if scales.size not in (1, 14) or not np.all(scales == 1.0):
        raise ValueError("Policy action scale must equal 1.0")
    home = np.asarray([float(value) for value in metadata.get("default_joint_pos", "nan").split(",")])
    if home.shape != (14,) or not np.allclose(home, HOME, atol=5e-4, rtol=0):
        raise ValueError("Policy HOME differs from the official pose")
    from oh_my_duck.rl.backends.isaac_newton.contracts import OBSERVATION_NAMES
    if tuple(metadata.get("observation_names", "").split(",")) != OBSERVATION_NAMES:
        raise ValueError("Policy observation order differs from the official order")
    return session, hashlib.sha256(path.read_bytes()).hexdigest()


class SimulationBackend:
    def __init__(self, *, robot_id: str, backend: str, policy_path: Path | None, task_id: str):
        if not robot_id:
            raise ValueError("robot_id is required")
        self.robot_id = robot_id
        self.backend = backend
        self.policy_path = policy_path.resolve() if policy_path is not None else None
        self.task_id = task_id
        self.policy, self.policy_sha256 = (
            validate_policy(self.policy_path) if self.policy_path is not None else (None, "")
        )
        self.identity = Identity(persona_id="microduck", robot_id=robot_id, domain=ExecutionDomain.SIMULATION)
        self.episode_id = uuid4().hex
        self._lock = asyncio.Lock()
        self._command = (0.0, 0.0, 0.0)
        self._expires_at = 0.0
        self._command_requests: dict[str, tuple[tuple[float, float, float], int]] = {}
        self._stopping = False
        self._sequence = 0
        self._simulation_time_s = 0.0
        self._measurements: dict | None = None
        self._sampled_at = ""
        self._tick_started_at = time.monotonic()
        self._tick_finished_at = self._tick_started_at
        self._active_task_id: str | None = None
        self._runner = None
        self._executor = None

    def bind_runner(self, runner) -> None:
        self._runner = runner

    def bind_executor(self, executor) -> None:
        self._executor = executor

    async def capabilities(self) -> RobotCapabilities:
        sensors = ("robot_state", "imu", "joint_state")
        if getattr(self, "catalog", None) is not None:
            sensors += ("head_rgb", "tof", "odometry")
            skills = ()
            command_limits = {"vx_m_s": [-0.4, 0.4], "vy_m_s": [-0.3, 0.3],
                              "yaw_rad_s": [-1.0, 1.0]}
        else:
            skills = ("move_for",)
            command_limits = {"vx_m_s": [-0.5, 0.5], "vy_m_s": [-0.5, 0.5],
                              "yaw_rad_s": [-2.0, 2.0], "ttl_ms": [1, 60000]}
        return RobotCapabilities(self.identity, self.backend, sensors=sensors,
            skills=skills, command_limits=command_limits)

    async def state(self) -> RobotState:
        async with self._lock:
            if self._measurements is None:
                raise RuntimeError("Simulation has not produced a measurement")
            values = self._measurements.copy()
            if not values["valid"] or values["fallen"]:
                motion_state = "invalid"
            elif self._is_moving(values):
                motion_state = "moving"
            else:
                motion_state = "stopped"
            return RobotState(self.identity, self._sampled_at, "simulation", motion_state,
                              self._active_task_id, self._simulation_time_s, self._sequence, values,
                              (f"simulation:{self.episode_id}:{self._sequence}",))

    async def read_sensor(self, sensor_id: str, *, max_age_ms: int) -> SensorFrame:
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or max_age_ms < 0:
            raise ValueError("max_age_ms must be a nonnegative integer")
        allowed = (await self.capabilities()).sensors
        if sensor_id not in allowed:
            raise ValueError(f"Unsupported sensor: {sensor_id}")
        async with self._lock:
            if self._measurements is None:
                raise RuntimeError("Simulation has not produced a measurement")
            age_ms = 1000 * (time.monotonic() - self._tick_finished_at)
            value = self._measurements.copy()
            if sensor_id == "imu":
                value = {key: value[key] for key in ("angular_velocity_rad_s", "projected_gravity")}
            if sensor_id == "joint_state":
                value = {key: value[key] for key in ("joint_names", "joint_position_rad", "joint_velocity_rad_s")}
            if sensor_id == "head_rgb":
                value = {key: value[key] for key in ("rgb_png_base64", "rgb_width", "rgb_height",
                                                     "camera_frame_id", "capture_time_s")}
            if sensor_id == "tof":
                value = {key: value[key] for key in ("tof_distance_mm", "tof_status", "tof_rows",
                                                     "tof_cols", "tof_frame_id", "capture_time_s")}
            if sensor_id == "odometry":
                value = value["odometry"].copy()
            value["simulation_time_s"] = self._simulation_time_s
            return SensorFrame(sensor_id, self.robot_id, self.episode_id, self._sequence,
                               self._tick_finished_at, "monotonic", _utc_now(),
                               {"imu": "trunk_base", "head_rgb": "head_camera", "tof": "tof",
                                "odometry": "world"}.get(sensor_id, "robot"),
                               Validity.VALID if age_ms <= max_age_ms else Validity.STALE,
                               None, readings=value,
                               evidence_refs=(f"simulation:{self.episode_id}:{self._sequence}",))

    async def command_velocity(self, vx_m_s: float, vy_m_s: float, yaw_rad_s: float,
                               *, request_id: str, ttl_ms: int, owner_task_id: str | None = None) -> None:
        velocity = _finite_velocity(vx_m_s, vy_m_s, yaw_rad_s)
        if not request_id or isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms < 1 or ttl_ms > 60000:
            raise ValueError("request_id and ttl_ms from 1 to 60000 are required")
        async with self._lock:
            if self._active_task_id is not None and owner_task_id != self._active_task_id:
                raise MotionBusyError("Motion is owned by an active skill")
            if owner_task_id is not None and owner_task_id != self._active_task_id:
                raise MotionBusyError("Motion owner does not match the active skill")
            existing = self._command_requests.get(request_id)
            if existing is not None:
                if existing != (velocity, ttl_ms):
                    raise ValueError("request_id was already used with different velocity parameters")
                return
            if self._stopping:
                raise MotionBusyError("Motion stop confirmation is in progress")
            self._command_requests[request_id] = (velocity, ttl_ms)
            self._command = velocity
            self._expires_at = time.monotonic() + ttl_ms / 1000

    async def stop_motion(self, *, request_id: str) -> StopResult:
        if not request_id:
            raise ValueError("request_id is required")
        async with self._lock:
            if self._stopping:
                raise MotionBusyError("Motion stop confirmation is in progress")
            self._stopping = True
            self._command = (0.0, 0.0, 0.0)
            self._expires_at = 0.0
            initial_sequence = self._sequence
        deadline = time.monotonic() + STOP_TIMEOUT_S
        samples = 0
        evidence: list[str] = []
        reason = "Measured motion did not settle within the confirmation window"
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(CONTROL_DT / 2)
                async with self._lock:
                    if self._sequence == initial_sequence:
                        continue
                    initial_sequence = self._sequence
                    values = self._measurements
                    evidence.append(f"simulation:{self.episode_id}:{self._sequence}")
                    if values is None or not values["valid"] or values["fallen"]:
                        reason = "Simulation state is invalid or the robot has fallen"
                        break
                    if self._is_moving(values):
                        samples = 0
                    else:
                        samples += 1
                    if samples >= STOP_SAMPLES:
                        return StopResult(request_id, self.robot_id, True, tuple(evidence[-STOP_SAMPLES:]))
            return StopResult(request_id, self.robot_id, False, tuple(evidence), reason)
        finally:
            async with self._lock:
                self._stopping = False

    @staticmethod
    def _is_moving(values: dict) -> bool:
        twist = values["body_twist"]
        return math.hypot(twist[0], twist[1]) > STOP_LINEAR_M_S or abs(twist[2]) > STOP_ANGULAR_RAD_S

    async def advance(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now >= self._expires_at:
                self._command = (0.0, 0.0, 0.0)
            command = self._command
            self._tick_started_at = now
            values = await asyncio.get_running_loop().run_in_executor(self._executor, self._physics_step, command)
            self._tick_finished_at = time.monotonic()
            self._sampled_at = _utc_now()
            self._sequence += 1
            self._simulation_time_s += CONTROL_DT
            values["command"] = list(command)
            values["valid"] = True
            values["step_wall_time_s"] = self._tick_finished_at - self._tick_started_at
            self._measurements = values

    async def run(self) -> None:
        target = time.monotonic()
        while True:
            await self.advance()
            target += CONTROL_DT
            await asyncio.sleep(max(0.0, target - time.monotonic()))

    def _physics_step(self, command: tuple[float, float, float]) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CpuMujocoBamBackend(SimulationBackend):
    def __init__(self, *, robot_id: str, policy_path: Path | None = None,
                 task_id: str = "Mjlab-Velocity-Flat-MicroDuck", catalog_dir: Path | None = None):
        import mujoco
        import numpy as np
        from oh_my_duck.rl.evaluation.rehearsal import infer_policy as ip

        if (policy_path is None) == (catalog_dir is None):
            raise ValueError("Specify exactly one of policy_path or catalog_dir")
        super().__init__(robot_id=robot_id, backend="cpu-mujoco-bam", policy_path=policy_path, task_id=task_id)
        if catalog_dir is None:
            from oh_my_duck.rl.tasks.recipes import build_environment
            from oh_my_duck.rl.training.tasks import project_tasks

            task = project_tasks().get(task_id)
            if task.model != "walk" or task.evaluation is None:
                raise ValueError("Interactive velocity execution requires a registered walking task")
            build_environment(task.binding("mujoco"), play=True)
        self._mujoco = mujoco
        self._np = np
        self.catalog = None
        if catalog_dir is not None:
            from oh_my_duck.robotics.microduck.official_policies import OfficialPolicyCatalogue

            self.catalog = OfficialPolicyCatalogue(catalog_dir)
            self.active_policy = self.catalog.get("velstand")
            self.policy_sha256 = self.active_policy.sha256
            scene = Path(ip.MICRODUCK_XML).with_name("scene_apartment.xml")
            policy_path = catalog_dir / self.active_policy.file
        else:
            scene = Path(ip.MICRODUCK_XML)
        if self.catalog is not None:
            source = mujoco.MjSpec.from_file(str(scene))
            self._source_target_ranges = np.asarray([act.ctrlrange for act in source.actuators], dtype=float)
            if self._source_target_ranges.shape != (14, 2) or not np.allclose(
                self._source_target_ranges, [-10.0, 10.0], atol=0, rtol=0
            ):
                raise ValueError("Official source position-target ranges differ")
        bam = ip.load_bam_model(200.0, 7.4, ip.BAM_MAX_CURRENT)
        self.model, self.data, self.controller, names = ip.load_mujoco_with_bam(
            str(scene), bam, 0.005, 0.1, ip.BAM_VIN_MIN)
        self.policy_inference = ip.PolicyInference(self.model, self.data,
            walking_onnx_path=str(policy_path), action_scale=1.0, bam_ctrl=self.controller,
            use_projected_gravity=True, new_cmd_obs=True)
        if tuple(names) != JOINT_NAMES or not np.allclose(self.policy_inference.default_pose, HOME, atol=1e-7, rtol=0):
            raise ValueError("CPU BAM joint order or HOME does not match the official policy")
        mujoco.mj_resetData(self.model, self.data)
        root = int(self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id])
        self.data.qpos[root:root + 7] = [0, 0, 0.125, 1, 0, 0, 0]
        self.data.qpos[self.policy_inference.joint_qpos_indices] = HOME
        self.controller.reset(self.data.qpos)
        self.policy_inference.set_position_targets(self.policy_inference.default_pose)
        mujoco.mj_forward(self.model, self.data)
        if self.catalog is not None:
            self._owner_thread = threading.get_ident()
            self._stop_requested = threading.Event()
            self._requested_command = {"twist": (0.0, 0.0, 0.0), "head": (0.0,) * 4,
                                       "body": (0.0,) * 6, "posture": "stand"}
            self._selected_at_s = 0.0
            self._pending_inference = None
            self._applied_requests: dict[str, dict] = {}
            self._used_action_requests: set[str] = set()
            self._stopped_samples = 0
            self._goal = None
            self._goal_held_ticks = 0
            self._goal_checked_sequence = -1
            self._sensor_rng = np.random.default_rng(0)
            self._renderer = None
            self._observer_renderer = None
            self._camera_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
            self._tof_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tof")
            if self._camera_site_id < 0 or self._tof_site_id < 0:
                raise ValueError("The apartment robot requires head_camera and tof frames")
            self.model.vis.global_.offwidth = 640
            self.model.vis.global_.offheight = 480

    def _require_owner(self) -> None:
        if self.catalog is None:
            raise RuntimeError("Official apartment control requires catalog_dir")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("MuJoCo control must run on its owner thread")

    def signal_stop(self) -> None:
        self._stop_requested.set()

    def list_policies(self) -> dict:
        self._require_owner()
        from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION

        return {"revision": OFFICIAL_REVISION, "scene": "scene_apartment.xml",
                "policies": self.catalog.describe_apartment()}

    def action_spec(self) -> dict:
        self._require_owner()
        home = self._np.asarray(HOME)
        minimum = (self._source_target_ranges[:, 0] - home) / self.active_policy.action_scale
        maximum = (self._source_target_ranges[:, 1] - home) / self.active_policy.action_scale
        return {"version": 1, "joint_names": list(JOINT_NAMES), "home": list(HOME),
                "action_scale": self.active_policy.action_scale, "control_hz": 50,
                "physics_dt_s": 0.005, "physics_steps_per_action": 4,
                "policy_name": self.active_policy.name, "policy_sha256": self.active_policy.sha256,
                "robot_model": "robot_allcollisions", "actuator": "XL330 M6 BAM",
                "minimum": minimum.tolist(), "maximum": maximum.tolist(),
                "source_target_range_rad": self._source_target_ranges.tolist(),
                "command_slots": {"twist": {"names": ["vx_m_s", "vy_m_s", "yaw_rad_s"],
                                             "units": ["m/s", "m/s", "rad/s"],
                                             "maximum_absolute": [0.4, 0.3, 1.0]},
                                  "head": {"names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
                                           "units": ["rad"] * 4, "maximum_absolute": [1.1, 1.1, 1.4, 0.31]},
                                  "body": {"names": ["x", "y", "z", "roll", "pitch", "yaw"],
                                           "units": ["m", "m", "m", "rad", "rad", "rad"],
                                           "maximum_absolute": [0.02, 0.02, 0.03] + [math.pi / 6] * 3}},
                "head_body_behavior_status": "unverified_in_official_apartment"}

    def set_command(self, command: dict, request_id: str) -> dict:
        self._require_owner()
        if not request_id or not isinstance(command, dict):
            raise ValueError("set_command requires request_id and command object")
        if set(command) - {"twist", "head", "body", "posture"}:
            raise ValueError("Unsupported command field")
        values = self._requested_command.copy()
        if "twist" in command:
            if not isinstance(command["twist"], (list, tuple)) or len(command["twist"]) != 3:
                raise ValueError("twist must have three values")
            twist = _finite_velocity(*command["twist"])
            if any(abs(value) > limit for value, limit in zip(twist, (0.4, 0.3, 1.0))):
                raise ValueError("twist exceeds the official walking task command ranges")
            values["twist"] = twist
        for key, limits in (("head", (1.1, 1.1, 1.4, 0.31)),
                            ("body", (0.02, 0.02, 0.03, math.pi / 6, math.pi / 6, math.pi / 6))):
            if key in command:
                items = command[key]
                if not isinstance(items, (list, tuple)) or len(items) != len(limits) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or abs(value) > limit for value, limit in zip(items, limits)
                ):
                    raise ValueError(f"{key} requires {len(limits)} bounded finite values")
                values[key] = tuple(float(value) for value in items)
        if "posture" in command:
            if command["posture"] not in {"sit", "stand"}:
                raise ValueError("posture must be sit or stand")
            values["posture"] = command["posture"]
        self.catalog.validate_command(self.active_policy, velocity=values["twist"],
                                      posture=values["posture"], head=values["head"], body=values["body"])
        self._requested_command = values
        return {"request_id": request_id, "effective_after_sequence": self._sequence,
                "command": {key: list(value) if isinstance(value, tuple) else value
                            for key, value in values.items()}}

    def _command_block(self):
        return self.catalog.command_block(
            self.active_policy, velocity=self._requested_command["twist"],
            posture=self._requested_command["posture"],
            elapsed_s=float(self.data.time) - self._selected_at_s,
            head=self._requested_command["head"], body=self._requested_command["body"])

    def infer_policy(self, command: dict | None = None) -> dict:
        self._require_owner()
        if command is not None:
            self.set_command(command, request_id=f"infer:{self.episode_id}:{self._sequence}")
        if self._stop_requested.is_set():
            raise RuntimeError("Motion stop was requested")
        elapsed = float(self.data.time) - self._selected_at_s
        duration = self.active_policy.duration_s
        if duration is not None and elapsed >= duration - 1e-9:
            self._pending_inference = None
            return {"episode_id": self.episode_id, "sequence": self._sequence,
                    "policy": self.active_policy.name, "policy_sha256": self.active_policy.sha256,
                    "complete": True, "reason": "manifest_duration_elapsed",
                    "duration_s": duration, "elapsed_s": elapsed}
        block = self._command_block()
        self.policy_inference.command = block
        observation = self.policy_inference.get_observations()
        action = self.catalog.infer(self.active_policy, observation)
        self._pending_inference = {"episode_id": self.episode_id, "sequence": self._sequence,
                                   "policy": self.active_policy.name, "action": action.copy()}
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "policy": self.active_policy.name, "policy_sha256": self.active_policy.sha256,
                "complete": False,
                "command_block": block.tolist(), "observation": observation.tolist(),
                "action": action.tolist()}

    def apply_policy_action(self, action: list[float], request_id: str,
                            expected_sequence: int | None = None,
                            should_stop: Callable[[], bool] | None = None) -> dict:
        self._require_owner()
        if not request_id:
            raise ValueError("request_id is required")
        existing = self._applied_requests.get(request_id)
        if existing is not None:
            if (list(action) != existing["executed_actions"] or
                    existing["episode_id"] != self.episode_id or
                    existing["policy_name"] != self.active_policy.name or
                    (expected_sequence is not None and expected_sequence != existing["action_sequence"])):
                raise ValueError("request_id was reused with another action")
            return existing
        if request_id in self._used_action_requests:
            raise ValueError("Interrupted or unexecuted request_id cannot be replayed")
        values = self._np.asarray(action, dtype=self._np.float32)
        if values.shape != (14,) or not self._np.isfinite(values).all():
            raise ValueError("Action must contain 14 finite offsets")
        ticket = self._pending_inference
        if (ticket is None or ticket["episode_id"] != self.episode_id or
                ticket["sequence"] != self._sequence or ticket["policy"] != self.active_policy.name or
                not self._np.array_equal(values, ticket["action"])):
            raise ValueError("Action does not match the current policy inference ticket")
        if expected_sequence is not None and expected_sequence != self._sequence:
            raise ValueError("Action sequence is stale")
        if self._stop_requested.is_set() or (should_stop is not None and should_stop()):
            self._pending_inference = None
            self._used_action_requests.add(request_id)
            raise MotionBusyError("Motion stop was requested before actuation")
        source_sequence = self._sequence
        executed = 0
        targets = self._np.asarray(HOME, dtype=self._np.float32) + values * self.active_policy.action_scale
        if (targets < self._source_target_ranges[:, 0]).any() or (targets > self._source_target_ranges[:, 1]).any():
            raise ValueError("Official action exceeds source position-target range")
        self.policy_inference.set_position_targets(targets)
        for _ in range(4):
            if self._stop_requested.is_set() or (should_stop is not None and should_stop()):
                break
            self.controller.update()
            self._mujoco.mj_step(self.model, self.data)
            executed += 1
            if not self._np.isfinite(self.data.qpos).all() or not self._np.isfinite(self.data.qvel).all():
                raise FloatingPointError("Nonfinite apartment physical state")
            for warning in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"):
                if self.data.warning[getattr(self._mujoco.mjtWarning, warning)].number:
                    raise FloatingPointError(warning)
        interrupted = executed != 4
        if executed:
            self.policy_inference.last_action = values.copy()
            self._sequence += 1
            self._simulation_time_s = float(self.data.time)
            self._sampled_at = _utc_now()
            self._tick_finished_at = time.monotonic()
            if executed == 4:
                self._update_control_evidence()
            else:
                self._stopped_samples = 0
                self._goal_held_ticks = 0
        self._pending_inference = None
        result = self.observe_control()
        result.update({"request_id": request_id, "executed_actions": values.tolist() if executed else [],
                       "raw_sim_steps": executed, "interrupted": interrupted,
                       "action_sequence": source_sequence,
                       "policy_name": self.active_policy.name,
                       "policy_sha256": self.active_policy.sha256})
        self._used_action_requests.add(request_id)
        if not interrupted:
            self._applied_requests[request_id] = result
        if len(self._applied_requests) > 10000:
            raise RuntimeError("Applied-action request history exceeded its session limit")
        return result

    def _native_state(self) -> dict:
        policy = self.policy_inference
        velocity = self._np.zeros(6)
        self._mujoco.mj_objectVelocity(self.model, self.data, self._mujoco.mjtObj.mjOBJ_BODY,
                                       policy.trunk_base_id, velocity, 1)
        root = int(self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id])
        root_velocity = int(self.model.jnt_dofadr[self.model.joint("trunk_base_freejoint").id])
        position = self.data.qpos[root:root + 3].copy()
        quaternion = self.data.qpos[root + 3:root + 7].copy()
        linear_world = self.data.qvel[root_velocity:root_velocity + 3].copy()
        linear_body = policy.quat_rotate_inverse(
            quaternion.astype(self._np.float32), linear_world.astype(self._np.float32))
        gravity = policy.get_projected_gravity()
        tilt = math.acos(float(self._np.clip(-gravity[2], -1.0, 1.0)))
        yaw = math.atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                         1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
        yaw_rate = float(self.data.qvel[root_velocity + 5])
        twist = [float(linear_body[0]), float(linear_body[1]), yaw_rate]
        state = {"body_position_m": position.tolist(), "body_quaternion_wxyz": quaternion.tolist(),
                 "body_twist": twist, "angular_velocity_rad_s": velocity[:3].tolist(),
                 "body_twist_world": [float(linear_world[0]), float(linear_world[1]), yaw_rate],
                 "projected_gravity": gravity.tolist(), "joint_names": list(JOINT_NAMES),
                 "joint_position_rad": self.data.qpos[policy.joint_qpos_indices].tolist(),
                 "joint_velocity_rad_s": self.data.qvel[policy.joint_qvel_indices].tolist(),
                 "height_m": float(position[2]), "tilt_rad": tilt,
                 "fallen": bool(position[2] < 0.06 or tilt > 1.3),
                 "odometry": {"x_m": float(position[0]), "y_m": float(position[1]),
                              "yaw_rad": yaw, "frame_id": "world", "source": "MuJoCo native pose"},
                 "native_contact_count": int(self.data.ncon)}
        if not all(self._np.isfinite(value).all() for value in
                   (position, quaternion, velocity, linear_body, gravity,
                    self.data.qpos, self.data.qvel)):
            raise FloatingPointError("Nonfinite native apartment measurement")
        return state

    def _sensor_frame(self) -> dict:
        from PIL import Image
        from oh_my_duck.robotics.microduck.sim_sensors import tof_frame

        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, width=320, height=240)
        self._renderer.update_scene(self.data, camera="head_camera")
        rgb = self._renderer.render().copy()
        if rgb.shape != (240, 320, 3) or self._np.ptp(rgb) == 0:
            raise ValueError("Head camera returned an invalid RGB frame")
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        if not self._np.isfinite(self.data.site_xpos[self._tof_site_id]).all() or not self._np.isfinite(
            self.data.site_xmat[self._tof_site_id]
        ).all():
            raise FloatingPointError("Invalid native ToF pose")
        distances, statuses = tof_frame(self.model, self.data, self._tof_site_id,
                                         rng=self._sensor_rng)
        return {"rgb_png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                "rgb_width": 320, "rgb_height": 240,
                "tof_distance_mm": distances, "tof_status": statuses,
                "tof_rows": 8, "tof_cols": 8, "camera_frame_id": "head_camera",
                "tof_frame_id": "tof", "capture_time_s": float(self.data.time)}

    def observe_control(self) -> dict:
        self._require_owner()
        self.policy_inference.command = self._command_block()
        observation = self.policy_inference.get_observations()
        if observation.shape != (61,) or not self._np.isfinite(observation).all():
            raise FloatingPointError("Invalid official policy observation")
        measurements = self._native_state()
        measurements.update(self._sensor_frame())
        measurements.update({"policy_observation": observation.tolist(),
                             "policy_action": self.policy_inference.last_action.tolist(),
                             "command_block": self.policy_inference.command.tolist(),
                             "controller_targets_rad": self.controller.q_target.tolist(),
                             "controller_model": "XL330 M6 BAM", "home_rad": list(HOME),
                             "valid": True,
                             "policy_name": self.active_policy.name,
                             "policy_sha256": self.active_policy.sha256})
        elapsed = float(self.data.time) - self._selected_at_s
        duration = self.active_policy.duration_s
        measurements["episode_terminated"] = bool(
            duration is not None and elapsed >= duration - 1e-9
            or self.active_policy.name == "alpha_walking" and measurements["fallen"]
        )
        measurements["episode_termination_reason"] = (
            "manifest_duration_elapsed" if duration is not None and elapsed >= duration - 1e-9
            else "walking_fall" if self.active_policy.name == "alpha_walking" and measurements["fallen"]
            else None
        )
        self._measurements = measurements
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "simulation_time_s": float(self.data.time), "observed_at": _utc_now(),
                "measurements": measurements}

    def reset_episode(self, seed: int, goal: dict, spawn_pose: dict | None = None) -> dict:
        self._require_owner()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if spawn_pose is not None and set(spawn_pose) != {"x_m", "y_m", "yaw_rad"}:
            raise ValueError("spawn_pose requires x_m, y_m and yaw_rad")
        self._mujoco.mj_resetData(self.model, self.data)
        root = int(self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id])
        if spawn_pose is None:
            x, y, yaw = 0.0, 0.0, 0.0
        else:
            x, y, yaw = (float(spawn_pose[key]) for key in ("x_m", "y_m", "yaw_rad"))
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError("spawn_pose values must be finite")
        self.data.qpos[root:root + 7] = [x, y, 0.125, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qpos[self.policy_inference.joint_qpos_indices] = HOME
        self.controller.reset(self.data.qpos)
        self.policy_inference.last_action.fill(0.0)
        self.policy_inference.set_position_targets(self.policy_inference.default_pose)
        self._mujoco.mj_forward(self.model, self.data)
        self._sensor_rng = self._np.random.default_rng(seed)
        self.episode_id = uuid4().hex
        self._sequence = 0
        self._simulation_time_s = 0.0
        self._selected_at_s = 0.0
        self._requested_command = {"twist": (0.0, 0.0, 0.0), "head": (0.0,) * 4,
                                   "body": (0.0,) * 6, "posture": "stand"}
        self._stop_requested.clear()
        self._pending_inference = None
        self._applied_requests.clear()
        self._used_action_requests.clear()
        self._stopped_samples = 0
        self.bind_goal(goal)
        return self.observe_control()

    def bind_goal(self, goal: dict) -> dict:
        self._require_owner()
        if not isinstance(goal, dict) or goal.get("kind") not in {"room", "dock", "object"}:
            raise ValueError("goal requires room, dock or object kind")
        kind = goal["kind"]
        if kind == "room":
            if set(goal) != {"kind", "room", "hold_ticks"} or goal["room"] not in {
                "kitchen", "living_room", "corridor", "bedroom", "office", "bathroom"
            }:
                raise ValueError("Room goal requires a named apartment room and hold_ticks")
        elif kind == "dock":
            if set(goal) != {"kind", "hold_ticks"}:
                raise ValueError("Dock goal requires hold_ticks")
        elif (set(goal) != {"kind", "object", "distance_m", "hold_ticks"} or
              goal["object"] not in {"obj_0", "obj_1", "obj_2", "obj_3", "ball_0", "ball_1", "ball_2"} or
              isinstance(goal["distance_m"], bool) or not isinstance(goal["distance_m"], (int, float)) or
              not math.isfinite(goal["distance_m"]) or goal["distance_m"] <= 0):
            raise ValueError("Object goal requires a native object name, positive distance_m and hold_ticks")
        hold = goal["hold_ticks"]
        if isinstance(hold, bool) or not isinstance(hold, int) or hold < 1 or hold > 500:
            raise ValueError("hold_ticks must be an integer from 1 to 500")
        self._goal = goal.copy()
        self._goal_held_ticks = 0
        self._goal_checked_sequence = -1
        return {"episode_id": self.episode_id, "goal": self._goal,
                "supported_check_ids": ["goal_reached"]}

    def _goal_measurement(self) -> tuple[bool, dict, dict, list[float]]:
        if self._goal is None:
            raise RuntimeError("No apartment goal is bound")
        robot = self.data.xpos[self.policy_inference.trunk_base_id].copy()
        kind = self._goal["kind"]
        target = None
        if kind == "room":
            def wall_position(name: str, axis: int) -> float:
                return float(self.data.geom_xpos[self.model.geom(name).id, axis])

            west = wall_position("wall_west", 0)
            east = wall_position("wall_east", 0)
            corridor_west = wall_position("wB_m", 0)
            corridor_east = wall_position("wC_m2", 0)
            south = wall_position("wall_south", 1)
            north = wall_position("wall_north", 1)
            kitchen_divider = wall_position("wA_e", 1)
            bedroom_divider = wall_position("wF", 1)
            bathroom_divider = wall_position("wG", 1)
            clearance = 0.2
            rooms = {
                "kitchen": (west + clearance, corridor_west - clearance,
                            kitchen_divider + clearance, north - clearance),
                "living_room": (west + clearance, corridor_west - clearance,
                                south + clearance, kitchen_divider - clearance),
                "corridor": (corridor_west + clearance, corridor_east - clearance,
                             south + clearance, north - clearance),
                "bedroom": (corridor_east + clearance, east - clearance,
                            bedroom_divider + clearance, north - clearance),
                "office": (corridor_east + clearance, east - clearance,
                           bathroom_divider + clearance, bedroom_divider - clearance),
                "bathroom": (corridor_east + clearance, east - clearance,
                             south + clearance, bathroom_divider - clearance),
            }
            left, right, bottom, top = rooms[self._goal["room"]]
            reached = bool(left <= robot[0] <= right and bottom <= robot[1] <= top)
            target = {"room": self._goal["room"], "bounds_xy_m": [left, right, bottom, top],
                      "source": "MuJoCo named wall geom positions", "clearance_m": clearance}
        elif kind == "dock":
            site = self.model.site("dock_site").id
            point = self.data.site_xpos[site].copy()
            distance = float(self._np.linalg.norm(robot[:2] - point[:2]))
            reached = distance <= 0.35
            target = {"site": "dock_site", "world_position_m": point.tolist(),
                      "distance_xy_m": distance, "threshold_m": 0.35}
        else:
            body = self.model.body(self._goal["object"]).id
            point = self.data.xpos[body].copy()
            distance = float(self._np.linalg.norm(robot[:2] - point[:2]))
            reached = distance <= self._goal["distance_m"]
            target = {"body": self._goal["object"], "world_position_m": point.tolist(),
                      "distance_xy_m": distance, "threshold_m": self._goal["distance_m"]}
        state = self._native_state()
        reached = reached and state["height_m"] >= 0.09 and state["tilt_rad"] <= math.radians(25)
        return reached, target, state, robot.tolist()

    def _update_control_evidence(self) -> None:
        state = self._native_state()
        self._stopped_samples = (
            self._stopped_samples + 1 if not state["fallen"] and not self._is_moving(state) else 0
        )
        reached, _, _, _ = self._goal_measurement()
        self._goal_held_ticks = self._goal_held_ticks + 1 if reached else 0

    def check_goal(self) -> dict:
        self._require_owner()
        _, target, state, robot = self._goal_measurement()
        complete = self._goal_held_ticks >= self._goal["hold_ticks"]
        evidence = {"robot_world_position_m": robot, "target": target,
                    "height_m": state["height_m"], "tilt_rad": state["tilt_rad"],
                    "upright_minimum_height_m": 0.09,
                    "upright_maximum_tilt_rad": math.radians(25),
                    "native_contact_count": state["native_contact_count"],
                    "held_ticks": self._goal_held_ticks,
                    "required_hold_ticks": self._goal["hold_ticks"]}
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "simulation_time_s": float(self.data.time),
                "checks": {"goal_reached": {"satisfied": complete,
                                            "reason": "Native target and upright hold condition",
                                            "evidence": evidence}}, "complete": complete}

    def select_policy(self, policy_name: str, request_id: str) -> dict:
        self._require_owner()
        if not request_id:
            raise ValueError("request_id is required")
        if self._pending_inference is not None:
            raise MotionBusyError("A policy action is awaiting execution")
        if self._stopped_samples < STOP_SAMPLES:
            raise MotionBusyError("Policy selection requires five measured stopped control samples")
        selected = self.catalog.get(policy_name)
        if selected.mode == "roller":
            raise ValueError("Roller policy requires a roller robot model; apartment uses allcollisions")
        self.active_policy = selected
        self.policy_sha256 = selected.sha256
        self._selected_at_s = float(self.data.time)
        self._requested_command = {"twist": (0.0, 0.0, 0.0), "head": (0.0,) * 4,
                                   "body": (0.0,) * 6, "posture": "stand"}
        self.policy_inference.last_action.fill(0.0)
        self._stop_requested.clear()
        self._stopped_samples = 0
        return {"request_id": request_id, "policy_name": selected.name, "kind": selected.kind,
                "encoding": selected.encoding, "duration_s": selected.duration_s,
                "ramp_s": selected.ramp_s, "unwind_s": selected.unwind_s,
                "entry_pose": selected.entry_pose, "slot": selected.slot, "chain": selected.chain,
                "command": selected.command,
                "sha256": selected.sha256, "action_spec": self.action_spec()}

    def discard_pending_inference(self) -> dict:
        self._require_owner()
        discarded = self._pending_inference is not None
        self._pending_inference = None
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "discarded": discarded, "confirmed_stopped": self._stopped_samples >= STOP_SAMPLES}

    def clear_stop(self) -> dict:
        self._require_owner()
        if self._stopped_samples < STOP_SAMPLES:
            raise MotionBusyError("Stop can clear only after five measured stopped control samples")
        self._stop_requested.clear()
        return {"episode_id": self.episode_id, "sequence": self._sequence,
                "confirmed_stopped": True}

    def capture_observer(self, *, include_segmentation: bool = False) -> dict:
        self._require_owner()
        from PIL import Image

        if self._observer_renderer is None:
            self._observer_renderer = self._mujoco.Renderer(self.model, width=640, height=480)
        camera = self._mujoco.MjvCamera()
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.data.xpos[self.policy_inference.trunk_base_id]
        camera.distance = 0.5
        camera.azimuth = 45.0
        camera.elevation = -45.0
        self._observer_renderer.update_scene(self.data, camera=camera)
        rgb = self._observer_renderer.render()
        if rgb.shape != (480, 640, 3) or rgb.dtype != self._np.uint8:
            raise ValueError("Observer camera produced an invalid RGB frame")
        output = BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        result = {"episode_id": self.episode_id, "sequence": self._sequence,
                  "simulation_time_s": float(self.data.time),
                  "capture_time_s": float(self.data.time),
                  "camera_frame_id": "observer_follow", "rgb_width": 640, "rgb_height": 480,
                  "rgb_png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                  "camera": {"mode": "free_follow_trunk", "lookat_world_m": camera.lookat.tolist(),
                             "distance_m": float(camera.distance), "azimuth_deg": float(camera.azimuth),
                             "elevation_deg": float(camera.elevation)}}
        if include_segmentation:
            self._observer_renderer.enable_segmentation_rendering()
            segmentation = self._observer_renderer.render()
            self._observer_renderer.disable_segmentation_rendering()
            geom = segmentation[:, :, 1] == self._mujoco.mjtObj.mjOBJ_GEOM.value
            robot_root = self.model.body_rootid[self.policy_inference.trunk_base_id]
            robot_geom_ids = self._np.flatnonzero(
                self.model.body_rootid[self.model.geom_bodyid] == robot_root)
            visible = geom & self._np.isin(segmentation[:, :, 0], robot_geom_ids)
            visible_geoms = self._np.unique(segmentation[:, :, 0][visible])
            apartment_visible = geom & ~visible
            body_ids = self._np.unique(self.model.geom_bodyid[visible_geoms])
            result["segmentation"] = {
                "visible_robot_geom_pixels": int(visible.sum()),
                "visible_robot_geom_count": int(visible_geoms.size),
                "visible_apartment_geom_pixels": int(apartment_visible.sum()),
                "visible_robot_body_names": [self._mujoco.mj_id2name(
                    self.model, self._mujoco.mjtObj.mjOBJ_BODY, int(body_id))
                    for body_id in body_ids],
            }
        return result

    def close(self) -> None:
        if self.catalog is not None and self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.catalog is not None and self._observer_renderer is not None:
            self._observer_renderer.close()
            self._observer_renderer = None

    def _physics_step(self, command: tuple[float, float, float]) -> dict:
        if self.catalog is not None:
            raise RuntimeError("Official apartment control requires the gated policy-action interface")
        mujoco = self._mujoco
        np = self._np
        policy = self.policy_inference
        policy.set_vel_cmd(*command)
        observation = policy.get_observations()
        if observation.shape != (61,) or not np.isfinite(observation).all():
            raise FloatingPointError("Invalid CPU BAM policy observation")
        action = policy.infer()
        if action.shape != (14,) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid CPU BAM policy action")
        policy.apply_action(action)
        for _ in range(4):
            self.controller.update()
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise FloatingPointError("Invalid CPU BAM physical state")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 policy.trunk_base_id, velocity, 1)
        root = int(self.model.jnt_qposadr[self.model.joint("trunk_base_freejoint").id])
        root_velocity = int(self.model.jnt_dofadr[self.model.joint("trunk_base_freejoint").id])
        quaternion = self.data.qpos[root + 3:root + 7].copy()
        linear_world = self.data.qvel[root_velocity:root_velocity + 3].copy()
        linear_body = policy.quat_rotate_inverse(quaternion.astype(np.float32),
                                                 linear_world.astype(np.float32))
        yaw_rate = float(self.data.qvel[root_velocity + 5])
        gravity = policy.get_projected_gravity()
        height = float(self.data.qpos[root + 2])
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        return {
            "body_twist": [float(linear_body[0]), float(linear_body[1]), yaw_rate],
            "body_twist_world": [float(linear_world[0]), float(linear_world[1]), yaw_rate],
            "angular_velocity_rad_s": [float(value) for value in velocity[:3]],
            "projected_gravity": gravity.tolist(),
            "joint_names": list(JOINT_NAMES),
            "joint_position_rad": self.data.qpos[policy.joint_qpos_indices].tolist(),
            "joint_velocity_rad_s": self.data.qvel[policy.joint_qvel_indices].tolist(),
            "height_m": height,
            "body_position_m": self.data.qpos[root:root + 3].tolist(),
            "body_quaternion_wxyz": quaternion.tolist(),
            "tilt_rad": tilt,
            "fallen": height < 0.06 or tilt > 1.3,
            "policy_observation": observation.tolist(),
            "policy_action": action.tolist(),
        }


class IsaacNewtonBackend(SimulationBackend):
    def __init__(self, *, robot_id: str, policy_path: Path, task_id: str, device: str):
        import torch
        from oh_my_duck.rl.backends.isaac_newton.contracts import joint_indices
        from oh_my_duck.rl.backends.isaac_newton.physics import inspect_solver
        from oh_my_duck.rl.tasks.recipes import build_environment
        from oh_my_duck.rl.training.runtime import create_environment
        from oh_my_duck.rl.training.tasks import project_tasks

        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("Isaac/Newton requires an explicitly selected CUDA device")
        super().__init__(robot_id=robot_id, backend="isaac-newton", policy_path=policy_path, task_id=task_id)
        task = project_tasks().get(task_id)
        if task.model != "walk" or task.evaluation is None:
            raise ValueError("Interactive velocity execution requires a registered walking task")
        cfg = build_environment(task.binding("isaac-newton"), play=True)
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        cfg.episode_length_s = 3600.0
        self._torch = torch
        self._joint_indices = joint_indices
        self.environment = create_environment(task, cfg, backend="isaac-newton", device=device)
        self.solver = inspect_solver(self.environment)
        self.observation, _ = self.environment.reset()
        if "actor" not in self.observation or tuple(self.observation["actor"].shape) != (1, 61):
            raise ValueError("Newton task did not produce a 61-dimensional actor observation")
        robot = self.environment.scene["robot"]
        self._joint_order = joint_indices(robot.joint_names)
        if abs(float(cfg.sim.mujoco.timestep * cfg.decimation) - CONTROL_DT) > 1e-10:
            raise ValueError("Newton task control interval differs from 50 Hz")

    def _physics_step(self, command: tuple[float, float, float]) -> dict:
        import numpy as np
        from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch

        torch = self._torch
        env = self.environment
        state = self.observation["actor"].clone()
        state[:, 48:] = 0.0
        state[:, 48:51] = torch.tensor(command, device=env.device)
        if tuple(state.shape) != (1, 61) or not torch.isfinite(state).all():
            raise FloatingPointError("Invalid Newton actor observation")
        for name in env.command_manager.active_terms:
            value = env.command_manager.get_command(name)
            value.zero_()
            if name == "twist":
                value[:] = torch.tensor(command, device=env.device)
        action = self.policy.run(None, {self.policy.get_inputs()[0].name: state.cpu().numpy()})[0]
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid Newton policy action")
        with torch.inference_mode():
            self.observation, reward, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
        if terminated.any() or truncated.any():
            raise RuntimeError("Newton task terminated; interactive session requires a valid measured state")
        if not torch.isfinite(reward).all() or not torch.isfinite(self.observation["actor"]).all():
            raise FloatingPointError("Invalid Newton physical state")
        robot = env.scene["robot"].data
        linear = as_torch(robot.root_link_lin_vel_b)[0].detach().cpu()
        angular = as_torch(robot.root_link_ang_vel_b)[0].detach().cpu()
        positions = as_torch(robot.joint_pos)[0, list(self._joint_order)].detach().cpu()
        velocities = as_torch(robot.joint_vel)[0, list(self._joint_order)].detach().cpu()
        gravity = as_torch(robot.projected_gravity_b)[0].detach().cpu()
        height = float((as_torch(robot.root_link_pos_w) - as_torch(env.scene.env_origins))[0, 2])
        tilt = math.acos(float((-gravity[2]).clamp(-1, 1)))
        return {
            "body_twist": [float(linear[0]), float(linear[1]), float(angular[2])],
            "angular_velocity_rad_s": angular.tolist(),
            "projected_gravity": gravity.tolist(),
            "joint_names": list(JOINT_NAMES),
            "joint_position_rad": positions.tolist(),
            "joint_velocity_rad_s": velocities.tolist(),
            "height_m": height,
            "body_position_m": as_torch(robot.root_link_pos_w)[0].detach().cpu().tolist(),
            "tilt_rad": tilt,
            "fallen": height < 0.06 or tilt > 1.3,
            "policy_observation": state[0].cpu().tolist(),
            "policy_action": action[0].tolist(),
        }

    def close(self) -> None:
        self.environment.close()
