from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from physical_harness.environments import (
    NativeCheck, NativeEnvironmentDescription, NativeFrame, NativeObservation, NativeStep,
)
from physical_harness.execution.modes import ExecutionMode
from physical_harness.execution.native_device import NativeActionDevice
from physical_harness.execution.worker import NativeWorkerSession, require_object
from physical_harness.policies.server import serve_policy
from physical_harness.validation import ContractValidator

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend, STOP_SAMPLES
from oh_my_duck.robotics.microduck.motion_guard import MotionGuard
from oh_my_duck.robotics.microduck.metric_motion import MetricMotion


def _wire_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class MicroDuckEnvironment:
    def __init__(self, configuration: Mapping[str, object]) -> None:
        self.configuration = dict(configuration)
        self.backend: CpuMujocoBamBackend | None = None
        self._description: NativeEnvironmentDescription | None = None
        self._last_sequence = -1
        self._last_control_started_at: float | None = None
        self._observations: dict[str, tuple[str, int]] = {}
        self._navigation_samples: dict[str, dict[str, Any]] = {}

    def _backend(self) -> CpuMujocoBamBackend:
        if self.backend is None:
            raise RuntimeError("MicroDuck scene has not been initialized")
        return self.backend

    def describe(self) -> NativeEnvironmentDescription:
        if self._description is None:
            raise RuntimeError("MicroDuck scene has not been described")
        return self._description

    def _observation(self, state: dict) -> NativeObservation:
        measured = state["measurements"]
        sequence = state["sequence"]
        if type(sequence) is not int or sequence < self._last_sequence:
            raise RuntimeError("MicroDuck observation sequence moved backward")
        self._last_sequence = sequence
        image = base64.b64decode(measured["rgb_png_base64"], validate=True)
        channels = {
            "policy_observation": tuple(measured["policy_observation"]),
            "joint_position_rad": tuple(measured["joint_position_rad"]),
            "joint_velocity_rad_s": tuple(measured["joint_velocity_rad_s"]),
            "body_position_m": tuple(measured["body_position_m"]),
            "body_twist": tuple(measured["body_twist"]),
            "imu": tuple(measured["angular_velocity_rad_s"] + measured["projected_gravity"]),
            "tof_distance_mm": tuple(measured["tof_distance_mm"]),
            "tof_status": tuple(measured["tof_status"]),
        }
        if any(not math.isfinite(item) for values in channels.values() for item in values):
            raise FloatingPointError("MicroDuck sensor state contains nonfinite values")
        observation_id = f"microduck:{state['episode_id']}:{sequence}:{uuid4().hex}"
        self._observations[observation_id] = (state["episode_id"], sequence)
        self._navigation_samples[observation_id] = {
            "episode_id": state["episode_id"], "sequence": sequence,
            "body_position_m": tuple(measured["body_position_m"]),
            "yaw_rad": float(measured["odometry"]["yaw_rad"]),
            "tof_distance_mm": tuple(measured["tof_distance_mm"]),
            "tof_status": tuple(measured["tof_status"]),
            "contact_evidence": measured["contact_evidence"],
        }
        if len(self._observations) > 128:
            oldest = next(iter(self._observations))
            self._observations.pop(oldest)
            self._navigation_samples.pop(oldest)
        return NativeObservation(
            observation_id=observation_id,
            observed_at=_wire_time(),
            observed_monotonic=time.monotonic(),
            images={"head_rgb": image},
            state=channels,
        )

    def reset(self, task_id: str, configuration: Mapping[str, object]) -> NativeObservation:
        if task_id != self.configuration["native_task_id"]:
            raise ValueError("MicroDuck task identity differs from the session scene")
        if self.backend is not None:
            raise RuntimeError("MicroDuck scene has already been initialized")
        catalog = Path(str(configuration["catalog_dir"])).resolve(strict=True)
        backend_name = configuration.get("backend", "cpu-mujoco-bam")
        if backend_name == "cpu-mujoco-bam":
            self.backend = CpuMujocoBamBackend(robot_id="microduck", catalog_dir=catalog)
        elif backend_name == "isaac-newton":
            from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend

            self.backend = IsaacNewtonBamBackend(
                robot_id="microduck", catalog_dir=catalog,
                scene_path=Path(str(configuration["usd_path"])).resolve(strict=True),
                device=str(configuration.get("device", "cuda:0")),
                scene_id=str(configuration["scene_id"]),
                robot_model=str(configuration.get("robot_model", "allcollisions")),
                observer_renderer=str(configuration.get("observer_renderer", "newton_warp")),
                provenance_path=Path(str(configuration["provenance_path"])).resolve(strict=True),
                public_map_path=(Path(str(configuration["public_map_path"])).resolve(strict=True)
                                 if "public_map_path" in configuration else None),
            )
        else:
            raise ValueError(f"Unknown MicroDuck simulation backend: {backend_name}")
        state = self.backend.reset_episode(
            seed=int(configuration["seed"]),
            goal=require_object(configuration["goal"]),
            spawn_pose=configuration.get("spawn_pose"),
        )
        specification = self.backend.action_spec()
        channels = [
            {"name": name, "quantity": "normalized", "unit": "dimensionless",
             "minimum": float(minimum), "maximum": float(maximum)}
            for name, minimum, maximum in zip(specification["joint_names"],
                                               specification["minimum"],
                                               specification["maximum"], strict=True)
        ]
        action_spec = {
            "schema_version": "physical.action_spec.v1",
            "embodiment_id": "microduck.xl330_bam",
            "version": f"{backend_name}-xl330-bam-policy-offset-v1",
            "coordinate_frame": "microduck.joint_order",
            "control_mode": "microduck.policy_offset",
            "frequency_hz": specification["control_hz"],
            "channels": channels,
        }
        self._description = NativeEnvironmentDescription(
            provider="microduck",
            embodiment_id="microduck.xl330_bam",
            action_spec=action_spec,
            camera_names=("head_rgb",),
            state_channels=("policy_observation", "joint_position_rad", "joint_velocity_rad_s",
                            "body_position_m", "body_twist", "imu", "tof_distance_mm",
                            "tof_status"),
            supported_check_ids=("goal_reached",),
            active_view_directions=(),
            task_instruction=str(configuration["task_instruction"]),
            scene_metadata={"scene": configuration.get("scene_id", "official_microduck_apartment"),
                            "backend": backend_name,
                            **({"dimensions_m": [8, 6]} if backend_name == "cpu-mujoco-bam" else
                               {"usd_path": configuration["usd_path"],
                                "provenance_path": configuration["provenance_path"]}),
                            "policy_revision": str(configuration["policy_revision"]),
                            "seed": configuration["seed"], "spawn_pose": configuration.get("spawn_pose")},
        )
        return self._observation(state)

    def bind_task(self, task_id: str) -> None:
        if task_id != self.configuration["native_task_id"]:
            raise ValueError("MicroDuck retained task identity differs")
        self._backend()

    def observe(self) -> NativeObservation:
        return self._observation(self._backend().observe_control())

    def step(self, action: Sequence[float], should_stop: Callable[[], bool],
             on_live_frame=None) -> NativeStep:
        backend = self._backend()
        if on_live_frame is not None:
            raise ValueError("MicroDuck live frame callback is unavailable")
        if self._last_control_started_at is not None:
            deadline = self._last_control_started_at + 1.0 / 50.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if should_stop():
                    return NativeStep(self.observe(), 0, False, 0, False)
                time.sleep(min(0.002, remaining))
        if should_stop():
            return NativeStep(self.observe(), 0, False, 0, False)
        self._last_control_started_at = time.monotonic()
        result = backend.apply_policy_action(list(action), request_id=uuid4().hex,
                                             expected_sequence=backend._sequence,
                                             should_stop=should_stop)
        raw_steps = result["raw_sim_steps"]
        if type(raw_steps) is not int or not 0 <= raw_steps <= 4:
            raise RuntimeError("MicroDuck returned an invalid physical step count")
        observation = self._observation(result)
        frames: tuple[NativeFrame, ...] = ()
        if raw_steps == 4 and result["sequence"] % 5 == 0:
            captured = backend.capture_observer()
            if ((captured["episode_id"], captured["sequence"]) !=
                    (result["episode_id"], result["sequence"]) or
                    captured["simulation_time_s"] != result["simulation_time_s"] or
                    captured["camera_frame_id"] != "observer_follow"):
                raise RuntimeError("Observer frame differs from its admitted physical step")
            frames = (NativeFrame(
                observation_id=f"microduck:observer:{captured['episode_id']}:{captured['sequence']}:{uuid4().hex}",
                observed_at=_wire_time(),
                images={"observer_follow": base64.b64decode(
                    captured["rgb_png_base64"], validate=True)},
                native_step_index=raw_steps,
                simulation_time_s=captured["simulation_time_s"],
            ),)
        return NativeStep(
            observation=observation,
            executed_actions=1 if raw_steps else 0,
            action_completed=raw_steps == 4,
            raw_sim_steps=raw_steps,
            episode_terminated=bool(result["measurements"]["episode_terminated"]),
            native_frames=frames,
        )

    def check(self, check_ids: Sequence[str]) -> Sequence[NativeCheck]:
        if list(check_ids) != ["goal_reached"]:
            raise ValueError("MicroDuck supports only goal_reached")
        result = self._backend().check_goal()
        check = result["checks"]["goal_reached"]
        return [NativeCheck("goal_reached", check["satisfied"],
                            json.dumps({"reason": check["reason"], "evidence": check["evidence"],
                                        "episode_id": result["episode_id"],
                                        "sequence": result["sequence"]}, allow_nan=False))]

    def turn_view(self, direction: str) -> NativeObservation:
        raise ValueError(f"MicroDuck does not support active view direction {direction}")

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


class MicroDuckActionDevice(NativeActionDevice):
    def __init__(self, environment: MicroDuckEnvironment,
                 publish_pausing: Callable[[NativeObservation], Any]) -> None:
        super().__init__(environment)
        self._publish_pausing = publish_pausing

    async def stop(self, execution_id: str, generation: int) -> dict:
        acknowledgement = await super().stop(execution_id, generation)
        observation = await self.on_owner(self.environment.observe)
        await self._publish_pausing(observation)
        return acknowledgement


class MicroDuckWorkerSession(NativeWorkerSession):
    DEFAULT_COMMAND_STEPS = 75
    MIN_COMMAND_STEPS = 5
    MAX_COMMAND_STEPS = 100

    def __init__(self, emit) -> None:
        super().__init__(emit)
        self._policy_server = None
        self._control_server = None
        self._control_secret: str | None = None
        self._motion_segment: dict[str, Any] | None = None
        self._motion_guard: dict[str, Any] | None = None
        self._motion_cleanup_task: asyncio.Task[None] | None = None
        self._guard_tof_boundary_id: str | None = None
        self._metric_motion: MetricMotion | None = None

    async def _await_motion_cleanup(self) -> None:
        task = self._motion_cleanup_task
        if task is not None:
            await task
            if self._motion_cleanup_task is task:
                self._motion_cleanup_task = None

    def _command_steps(self, arguments: dict[str, Any]) -> int:
        count = arguments.get("max_control_steps", self.DEFAULT_COMMAND_STEPS)
        if type(count) is not int or not self.MIN_COMMAND_STEPS <= count <= self.MAX_COMMAND_STEPS:
            raise ValueError("MicroDuck command requires 5 to 100 control steps")
        return count

    def _bind_motion_segment(self, command_result: dict[str, Any], count: int,
                             preserve_guard: bool = False) -> None:
        self._motion_guard = None
        self._guard_tof_boundary_id = None
        gate = self._gate
        snapshot = gate.snapshot() if gate is not None else None
        guard = (self._motion_segment["guard"] if preserve_guard else
                 MotionGuard(command_result["command"], command_result["effective_after_sequence"], count))
        if preserve_guard:
            guard.update_command(command_result["command"], command_result["effective_after_sequence"], count)
        self._motion_segment = {
            "run_task_id": self._run_task_id,
            "execution_id": None if snapshot is None or snapshot["state"] == "ended"
            else snapshot["execution_id"],
            "generation": None if snapshot is None or snapshot["state"] == "ended"
            else snapshot["generation"],
            "boundary_id": None if snapshot is None or snapshot["state"] == "ended"
            else snapshot["boundary_id"],
            "effective_after_sequence": command_result["effective_after_sequence"],
            "max_control_steps": count,
            "command": command_result["command"],
            "request_id": command_result["request_id"],
            "guard": guard,
        }

    def _guard_for_sample(self, sample: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any] | None:
        segment = self._motion_segment
        if segment is None or segment["run_task_id"] != self._run_task_id:
            raise RuntimeError("MicroDuck command segment lacks the active task identity")
        if segment["execution_id"] is None:
            segment["execution_id"] = snapshot["execution_id"]
            segment["generation"] = snapshot["generation"]
        if (segment["execution_id"] != snapshot["execution_id"] or
                segment["generation"] != snapshot["generation"]):
            raise RuntimeError("MicroDuck command segment belongs to another action generation")
        evidence = segment["guard"].observe(sample)
        if evidence is None:
            return None
        return {
            "run_task_id": self._run_task_id, "execution_id": snapshot["execution_id"],
            "generation": snapshot["generation"], "command_request_id": segment["request_id"],
            **evidence,
        }

    async def _complete_motion_pause(self, ready: asyncio.Event, gate, policy) -> None:
        await ready.wait()
        pump = self._pump
        if pump is not None:
            await pump
            if self._pump is pump:
                self._pump = None
        snapshot = gate.snapshot()
        if snapshot["state"] == "pausing":
            snapshot = await gate.pause(snapshot["stop_reason"],
                                        terminal=snapshot["stop_reason"] == "user_stop")
        if (self._gate is not gate or snapshot["state"] not in ("paused", "ended") or
                not snapshot["device_confirmed"]):
            raise RuntimeError(f"MicroDuck guarded pause lost its confirmed execution boundary: {snapshot}")
        if self._host_connected:
            await self._require_device().on_owner(
                self._environment._backend().discard_pending_inference)
            observation = await self._require_device().on_owner(self._environment.observe)
            await self._publish(observation, self._last_control)
        if self._policy is policy and policy is not None:
            await policy.close()
            self._policy = None

    def _require_control_lease(self, run_task_id: str) -> None:
        if run_task_id != self._run_task_id or not self._lease_active or not self._host_connected:
            raise RuntimeError("MicroDuck tool request lacks the active task lease")

    async def initialize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._device is not None:
            raise RuntimeError("MicroDuck worker is already initialized")
        if arguments["provider"] != "microduck":
            raise ValueError("MicroDuck worker requires its native provider")
        configuration = require_object(arguments["scene_configuration"])
        if configuration["native_task_id"] != arguments["native_task_id"]:
            raise ValueError("Native task identity differs from scene configuration")
        if arguments["execution_mode"] != "policy":
            raise ValueError("MicroDuck requires the learned-policy execution mode")
        self._provider = "microduck"
        self._native_task_id = arguments["native_task_id"]
        self._policy_id = arguments["policy_id"]
        self._execution_mode = ExecutionMode.POLICY
        self._control_mode = "0-shot"
        self._monitor_every_actions = arguments.get("monitor_every_actions", 1)
        self._policy_max_actions_per_inference = arguments.get("policy_max_actions_per_inference", 1)
        if self._monitor_every_actions != 1 or self._policy_max_actions_per_inference != 1:
            raise ValueError("MicroDuck requires one action and one update per control step")
        self._validator = ContractValidator.from_path(arguments["schema_path"])
        environment = MicroDuckEnvironment(configuration)
        self._environment = environment
        self._device = MicroDuckActionDevice(environment, self._publish_pausing)
        self._initial_observation = await self._device.on_owner(
            lambda: environment.reset(self._native_task_id, configuration))
        self._description = await self._device.on_owner(environment.describe)
        self._validator.parse("ActionSpec", self._description.action_spec)
        initial_check = await self._device.on_owner(environment._backend().check_goal)
        if initial_check["checks"]["goal_reached"]["satisfied"]:
            raise RuntimeError("MicroDuck goal is already satisfied at reset")
        initial_spatial = await self._device.on_owner(environment._backend()._goal_measurement)
        if initial_spatial[0]:
            raise RuntimeError("MicroDuck begins inside the native goal target")

        async def infer(request: dict[str, Any]) -> list[list[float]]:
            if request["observation_id"] != request["observation"]["source_observation_id"]:
                raise ValueError("Policy observation identity differs from EDH ticket")
            expected = environment._observations.get(request["observation_id"])
            if expected is None:
                raise ValueError("Policy observation identity was not issued by this environment")
            gate = self._require_gate()

            def inference_state() -> str:
                snapshot = gate.snapshot()
                if snapshot["execution_id"] != request["execution_id"]:
                    raise ValueError("Policy inference belongs to another execution")
                if (snapshot["state"] == "running" and
                        snapshot["generation"] == request["generation"]):
                    return "running"
                if (snapshot["state"] in ("pausing", "paused", "ended") and
                        request["generation"] < snapshot["generation"]):
                    return "cancelled"
                raise ValueError("Policy inference has an invalid action generation")

            if inference_state() == "cancelled":
                raise asyncio.CancelledError
            def infer_on_owner() -> dict:
                if inference_state() == "cancelled":
                    return {"cancelled": True}
                return environment._backend().infer_policy()

            result = await self._device.on_owner(infer_on_owner)
            if inference_state() == "cancelled":
                raise asyncio.CancelledError
            if (result["episode_id"], result["sequence"]) != expected:
                raise ValueError("Policy inference does not match its observed physical state")
            if result["complete"]:
                raise RuntimeError("Policy completed outside a physical termination boundary")
            return [result["action"]]

        self._policy_server = await serve_policy(infer, self._validator, port=0)
        self._policy_uri = f"ws://127.0.0.1:{self._policy_server.sockets[0].getsockname()[1]}"
        self._control_secret = str(configuration["control_secret"])
        if len(self._control_secret) != 64:
            raise ValueError("MicroDuck control secret must contain 64 hexadecimal characters")
        self._control_server = await asyncio.start_server(
            self._control, host="127.0.0.1", port=configuration["control_port"])
        return {
            "provider": self._description.provider,
            "embodiment_id": self._description.embodiment_id,
            "action_spec": self._description.action_spec,
            "camera_names": self._description.camera_names,
            "supported_check_ids": self._description.supported_check_ids,
            "active_view_directions": self._description.active_view_directions,
            "task_instruction": self._description.task_instruction,
            "scene_metadata": self._description.scene_metadata,
            "native_task_id": self._native_task_id,
            "clock_id": self._clock_id,
            "policy_id": self._policy_id,
            "execution_mode": self._execution_mode.value,
        }

    async def _control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line or len(line) > 65536:
                raise ValueError("MicroDuck tool request is empty or too large")
            request = require_object(json.loads(line))
            if not secrets.compare_digest(str(request["control_secret"]), self._control_secret):
                raise RuntimeError("MicroDuck control request lacks the session capability")
            run_task_id = request["run_task_id"]
            self._require_control_lease(run_task_id)
            operation = request["operation"]
            args = require_object(request.get("arguments", {}))
            backend = self._environment._backend()
            if operation in ("walk", "rotate"):
                async with self._control_lock:
                    await self._await_motion_cleanup()
                    self._require_control_lease(run_task_id)
                    snapshot = self._require_gate().snapshot()
                    if snapshot["state"] != "paused" or not snapshot["device_confirmed"]:
                        raise RuntimeError("Metric motion requires a confirmed paused native execution")
                    if self._motion_guard is not None and self._motion_guard["reason"] in (
                            "forward_proximity", "external_contact", "motion_stalled", "tof_invalid"):
                        raise RuntimeError("Resolve the measured motion hazard before metric navigation")
                    def prepare_metric():
                        self._require_control_lease(run_task_id)
                        if backend._stopped_samples < STOP_SAMPLES:
                            raise RuntimeError("Metric motion requires five measured stopped samples")
                        measured = backend._native_state()
                        motion = MetricMotion(operation, args, measured,
                            robot_model=getattr(backend, "robot_model", "allcollisions"))
                        motion.request_id = request["request_id"]
                        locomotion = ("roller" if getattr(backend, "robot_model", "allcollisions") == "groundcontact_rollers"
                                      else "alpha_walking")
                        selected = ({"policy_name": locomotion} if backend.active_policy.name == locomotion
                                    else backend.select_policy(locomotion, request["request_id"] + ":policy"))
                        command = backend.set_command(motion.command(measured), request["request_id"])
                        return motion, selected, command
                    motion, selected, command = await self._device.on_owner(prepare_metric)
                    self._metric_motion = motion
                    self._bind_motion_segment(command, self.MAX_COMMAND_STEPS)
                    result = {"prepared": True, "operation": operation, "arguments": args,
                              "request_id": request["request_id"], "command_admission": command,
                              "policy_name": selected["policy_name"],
                              "start_position_m": motion.start_position, "start_yaw_rad": motion.start_yaw,
                              "distance_tolerance_m": motion.DISTANCE_TOLERANCE_M,
                              "angle_tolerance_deg": motion.ANGLE_TOLERANCE_DEG,
                              "execution_id": snapshot["execution_id"],
                              "generation": snapshot["generation"], "boundary_id": snapshot["boundary_id"],
                              "next_action": "execution.resume"}
            elif operation == "set_command":
                async with self._control_lock:
                    await self._await_motion_cleanup()
                    self._require_control_lease(run_task_id)
                    count = self._command_steps(args)
                    snapshot = self._gate.snapshot() if self._gate is not None else None
                    if snapshot is not None and snapshot["state"] == "ended":
                        snapshot = None
                    if snapshot is not None and (snapshot["state"] != "paused" or
                                                 not snapshot["device_confirmed"]):
                        raise RuntimeError("Command adjustment requires a confirmed paused boundary")
                    hazard = self._motion_guard
                    if (hazard is not None and hazard["reason"] in
                            ("forward_proximity", "external_contact", "motion_stalled", "tof_invalid")):
                        current_twist = self._motion_segment["command"]["twist"]
                        next_twist = args["command"].get("twist", current_twist)
                        if any(abs(value) > 1e-6 for value in next_twist):
                            if list(next_twist) == hazard["command"]["twist"]:
                                raise RuntimeError("Guarded motion requires a changed twist command")
                            if (snapshot is None or
                                    self._guard_tof_boundary_id != snapshot["boundary_id"]):
                                raise RuntimeError("Guarded motion requires fresh ToF at this stopped boundary")
                    def apply_command() -> dict:
                        self._require_control_lease(run_task_id)
                        if snapshot is not None:
                            current = self._gate.snapshot()
                            if (current["execution_id"] != snapshot["execution_id"] or
                                    current["generation"] != snapshot["generation"] or
                                    current["boundary_id"] != snapshot["boundary_id"] or
                                    current["state"] != "paused"):
                                raise RuntimeError("Command boundary changed before physical owner mutation")
                        return backend.set_command(args["command"], request_id=request["request_id"])
                    result = await self._device.on_owner(apply_command)
                    self._metric_motion = None
                    self._bind_motion_segment(result, count)
                    result["max_control_steps"] = count
            elif operation in ("select_policy", "transition_policy"):
                async with self._control_lock:
                    await self._await_motion_cleanup()
                    self._require_control_lease(run_task_id)
                    def select_on_owner() -> dict:
                        self._require_control_lease(run_task_id)
                        if self._gate is not None:
                            snapshot = self._gate.snapshot()
                            if snapshot["state"] not in ("paused", "ended") or not snapshot["device_confirmed"]:
                                raise RuntimeError("Policy selection requires a confirmed stop boundary")
                        if operation == "transition_policy" and (
                                self._gate is None or snapshot["state"] != "ended" or
                                snapshot["stop_reason"] != "episode_terminated"):
                            raise RuntimeError("Policy transition requires a confirmed native episode termination")
                        policy = backend.catalog.get(args["policy_name"])
                        if policy.action_scale != backend.active_policy.action_scale:
                            raise ValueError("Selected policy changes the admitted physical action scale")
                        if self._gate is not None:
                            backend.discard_pending_inference()
                        select = (backend.transition_completed_policy if operation == "transition_policy"
                                  else backend.select_policy)
                        selected = select(args["policy_name"], request_id=request["request_id"])
                        specification = selected["action_spec"]
                        channels = [
                            {"name": name, "quantity": "normalized", "unit": "dimensionless",
                             "minimum": float(minimum), "maximum": float(maximum)}
                            for name, minimum, maximum in zip(specification["joint_names"],
                                                               specification["minimum"],
                                                               specification["maximum"], strict=True)
                        ]
                        if channels != self._description.action_spec["channels"]:
                            raise RuntimeError("Selected policy changes the admitted physical action semantics")
                        return selected
                    result = await self._device.on_owner(select_on_owner)
                    self._metric_motion = None
            elif operation == "read_sensor":
                if args["sensor"] not in ("head_rgb", "tof", "imu", "joint_state", "odometry"):
                    raise ValueError("Unknown MicroDuck sensor")
                await self._await_motion_cleanup()
                observed = await self._device.on_owner(backend.observe_control)
                self._require_control_lease(run_task_id)
                measured = observed["measurements"]
                if args["sensor"] == "tof" and self._motion_guard is not None:
                    snapshot = self._gate.snapshot() if self._gate is not None else None
                    if (snapshot is not None and snapshot["state"] == "paused" and
                            snapshot["device_confirmed"] and
                            snapshot["execution_id"] == self._motion_guard["execution_id"] and
                            observed["sequence"] >= self._motion_guard["sequence"]):
                        self._guard_tof_boundary_id = snapshot["boundary_id"]
                fields = {
                    "head_rgb": ("rgb_png_base64", "rgb_width", "rgb_height", "camera_frame_id", "camera_geometry"),
                    "tof": ("tof_distance_mm", "tof_status", "tof_rows", "tof_cols", "tof_frame_id"),
                    "imu": ("angular_velocity_rad_s", "projected_gravity"),
                    "joint_state": ("joint_names", "joint_position_rad", "joint_velocity_rad_s"),
                    "odometry": ("odometry", "body_twist"),
                }[args["sensor"]]
                result = {"sensor": args["sensor"], "episode_id": observed["episode_id"],
                          "sequence": observed["sequence"], "observed_at": observed["observed_at"],
                          "measurements": {key: measured[key] for key in fields},
                          "motion_guard": self._motion_guard}
                if args["sensor"] == "joint_state" and "passive_joints" in measured:
                    result["measurements"]["passive_joints"] = measured["passive_joints"]
            elif operation == "inspect_scene":
                await self._await_motion_cleanup()
                snapshot = self._require_gate().snapshot()
                if snapshot["state"] not in ("paused", "ended") or not snapshot["device_confirmed"]:
                    raise RuntimeError("Perception requires a confirmed paused execution boundary")
                prompt = args["prompt"]
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 120:
                    raise ValueError("Perception prompt must contain 1–120 characters")
                if args["source"] == "simulator_ground_truth":
                    result = await self._device.on_owner(lambda: backend.inspect_scene(prompt))
                elif args["source"] == "models":
                    from oh_my_duck.perception.client import PerceptionClient
                    endpoint = self._environment.configuration["perception_endpoint"]
                    frame = await self._device.on_owner(backend.perception_frame)
                    result = await asyncio.to_thread(PerceptionClient(endpoint).inspect, frame, prompt)
                else:
                    raise ValueError("Unknown perception source")
            elif operation == "progress":
                def progress_on_owner():
                    observation = backend.observe_control()
                    observation["measurements"]["stopped_samples"] = backend._stopped_samples
                    return observation
                observed = await self._device.on_owner(progress_on_owner)
                measured = observed["measurements"]
                gate_state = self._require_gate().snapshot() if self._gate is not None else None
                result = {"episode_id": observed["episode_id"], "sequence": observed["sequence"],
                          "simulation_time_s": observed["simulation_time_s"],
                          "policy_name": measured["policy_name"],
                          "body_position_m": measured["body_position_m"],
                          "body_twist": measured["body_twist"], "fallen": measured["fallen"],
                          "height_m": measured["height_m"], "tilt_rad": measured["tilt_rad"],
                          "stopped_samples": measured["stopped_samples"],
                          "required_stopped_samples": STOP_SAMPLES,
                          "metric_motion": None if self._metric_motion is None else self._metric_motion.result,
                          "command_block": measured["command_block"],
                          "contact_evidence": measured["contact_evidence"],
                          "motion_guard": self._motion_guard,
                          "command_segment": None if self._motion_segment is None else {
                              "request_id": self._motion_segment["request_id"],
                              "effective_after_sequence": self._motion_segment["effective_after_sequence"],
                              "max_control_steps": self._motion_segment["max_control_steps"],
                              "used_control_steps": max(0, observed["sequence"] -
                                                        self._motion_segment["effective_after_sequence"]),
                          },
                          "execution": None if gate_state is None else {
                              "execution_id": gate_state["execution_id"],
                              "generation": gate_state["generation"],
                              "state": gate_state["state"],
                              "device_confirmed": gate_state["device_confirmed"],
                              "boundary_id": gate_state["boundary_id"],
                          }}
            elif operation == "catalog":
                result = await self._device.on_owner(backend.list_policies)
                for policy in result["policies"]:
                    policy["executable_in_native_edh"] = policy["executable_here"]
                    if (policy["name"] == "alpha_walking" and
                            self._environment.configuration.get("backend", "cpu-mujoco-bam") ==
                            "cpu-mujoco-bam"):
                        policy["apartment_command_calibration"] = {
                            "pure_lateral_and_in_place_yaw": "weak_response_in_cpu_apartment",
                            "coupled_forward_lateral": "measured_motion_with_positive_vx_and_vy",
                            "coupled_reverse_lateral": "measured_retreat_with_negative_vx_and_positive_vy",
                            "coupled_forward_lateral_yaw": "measured_turning_motion_with_positive_vx_vy_yaw",
                            "tested_twists_m_s_rad_s": [[0.2, 0.25, 0.0], [-0.2, 0.3, 0.0],
                                                        [0.2, 0.25, 0.4]],
                            "scope": "one_deterministic_cpu_apartment_seed",
                        }
            elif operation == "finish_policy":
                result = await self._finish_policy(args, request["run_task_id"])
            elif operation == "scene":
                geometry = await self._device.on_owner(backend.public_scene_info)
                if self._environment.configuration.get("backend", "cpu-mujoco-bam") == "isaac-newton":
                    result = geometry
                else:
                    result = {
                        "scene": "scene_apartment.xml",
                        "frame": "world",
                        "direction": "+x east, +y north",
                        "rooms": ["kitchen", "living_room", "corridor", "bedroom", "office", "bathroom"],
                        "connections": [
                            {"from": "corridor", "to": "office",
                             "route": "east doorway slightly north of corridor center"},
                            {"from": "corridor", "to": "bedroom",
                             "route": "east doorway near the north end"},
                            {"from": "corridor", "to": "bathroom",
                             "route": "east doorway near the south end"},
                            {"from": "corridor", "to": "kitchen",
                             "route": "west doorway near the north end"},
                            {"from": "corridor", "to": "living_room",
                             "route": "west doorway near the south end"},
                        ],
                        "public_geometry": geometry,
                    }
            else:
                raise ValueError("Unknown MicroDuck tool operation")
            self._require_control_lease(run_task_id)
            writer.write((json.dumps({"result": result}, allow_nan=False) + "\n").encode())
            await writer.drain()
        except Exception as error:
            writer.write((json.dumps({"error": {"type": type(error).__name__,
                                                "message": str(error)}}) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _finish_policy(self, arguments: dict[str, Any], run_task_id: str) -> dict[str, Any]:
        async with self._control_lock:
            await self._await_motion_cleanup()
            if run_task_id != self._run_task_id or not self._lease_active:
                raise RuntimeError("Policy finish lacks the active task lease")
            gate = self._require_gate()
            snapshot = gate.snapshot()
            if snapshot["state"] != "paused" or not snapshot["device_confirmed"]:
                raise RuntimeError("Policy finish requires a confirmed paused execution")
            if (arguments["execution_id"] != snapshot["execution_id"] or
                    arguments["generation"] != snapshot["generation"] or
                    arguments["boundary_id"] != snapshot["boundary_id"]):
                raise RuntimeError("Policy finish identity differs from the confirmed boundary")
            segment = self._motion_segment
            if (segment is None or segment["run_task_id"] != run_task_id or
                    segment["execution_id"] != snapshot["execution_id"] or
                    any(abs(value) > 1e-6 for value in segment["command"]["twist"])):
                raise RuntimeError("Policy finish requires the current zero-twist command segment")
            def confirm_physical_stop() -> dict[str, Any]:
                self._require_control_lease(run_task_id)
                current = gate.snapshot()
                if (current["execution_id"] != snapshot["execution_id"] or
                        current["generation"] != snapshot["generation"] or
                        current["boundary_id"] != snapshot["boundary_id"] or
                        current["state"] != "paused" or not current["device_confirmed"]):
                    raise RuntimeError("Policy finish boundary changed before physical confirmation")
                backend = self._environment._backend()
                completed_zero_steps = backend._sequence - segment["effective_after_sequence"]
                state = backend._native_state()
                if completed_zero_steps < STOP_SAMPLES:
                    raise RuntimeError("Policy finish requires five executed zero-command steps")
                guard = self._motion_guard
                if (guard is None or guard["command_request_id"] != segment["request_id"] or
                        guard["run_task_id"] != run_task_id or
                        any(abs(value) > 1e-6 for value in backend._requested_command["twist"]) or
                        backend._stopped_samples < STOP_SAMPLES or
                        state["fallen"] or backend._is_moving(state)):
                    raise RuntimeError("Policy finish requires measured physical stop")
                return {"zero_control_steps": completed_zero_steps,
                        "stopped_samples": backend._stopped_samples,
                        "body_twist": state["body_twist"]}
            stop_confirmation = await self._require_device().on_owner(confirm_physical_stop)
            self._require_control_lease(run_task_id)
            await gate.pause("policy_stop", terminal=True)
            if self._pump is not None:
                await self._pump
                self._pump = None
            await self._require_device().on_owner(
                self._environment._backend().discard_pending_inference)
            observation = await self._require_device().on_owner(self._environment.observe)
            publication = await self._publish(observation)
            if self._policy is not None:
                await self._policy.close()
                self._policy = None
            return {"accepted": True, "run_task_id": run_task_id,
                    "execution": publication["status"],
                    "observation_id": observation.observation_id,
                    "stop_confirmation": stop_confirmation}

    async def open_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._await_motion_cleanup()
        state = await self._device.on_owner(self._environment._backend()._goal_measurement)
        if state[0]:
            raise RuntimeError("Retained MicroDuck scene already occupies this task goal")
        opened = await super().open_task(arguments)
        run_task_id = opened["run_task_id"]
        def initial_command() -> dict:
            self._require_control_lease(run_task_id)
            return self._environment._backend().set_command(
                {"twist": [0.0, 0.0, 0.0]}, request_id=uuid4().hex)
        command = await self._device.on_owner(initial_command)
        self._require_control_lease(run_task_id)
        self._bind_motion_segment(command, self.DEFAULT_COMMAND_STEPS)
        self._motion_guard = None
        self._guard_tof_boundary_id = None
        self._metric_motion = None
        return opened

    async def start(self, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._await_motion_cleanup()
        if self._motion_segment is None or self._motion_segment["run_task_id"] != self._run_task_id:
            raise RuntimeError("Native start requires a bounded MicroDuck command")
        return await super().start(arguments)

    async def resume(self, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._await_motion_cleanup()
        snapshot = self._require_gate().snapshot()
        segment = self._motion_segment
        if (segment is None or segment["run_task_id"] != self._run_task_id or
                segment["execution_id"] != snapshot["execution_id"] or
                segment["generation"] != snapshot["generation"] or
                segment["boundary_id"] != snapshot["boundary_id"] or
                snapshot["state"] != "paused" or not snapshot["device_confirmed"]):
            raise RuntimeError("Resume requires a new bounded command at the confirmed boundary")
        return await super().resume(arguments)

    async def close_task(self) -> dict[str, Any]:
        self._lease_active = False
        await self._await_motion_cleanup()
        return await super().close_task()

    async def _on_segment(self, segment: dict[str, Any], receipt: dict[str, Any]) -> None:
        await super()._on_segment(segment, receipt)
        gate = self._require_gate()
        snapshot = gate.snapshot()
        if (snapshot["state"] != "running" or
                self._require_device().executed_actions >= self._request["budget"]["max_control_steps"] or
                gate.remaining_wall_time() <= 0):
            return
        step = self._require_device().last_step
        sample = self._environment._navigation_samples[step.observation.observation_id]
        guard = self._guard_for_sample(sample, snapshot)
        motion = self._metric_motion
        if motion is not None:
            backend = self._environment._backend()
            state, stopped = await self._device.on_owner(lambda: (backend._native_state(), backend._stopped_samples))
            previous_phase = motion.phase
            evidence = motion.observe(sample, state, stopped)
            self._last_control["metric_motion"] = evidence
            if guard is not None and guard["reason"] != "command_segment_complete":
                motion.phase = "blocked"
                evidence.update(phase="blocked", completed=False, reason=guard["reason"])
            elif motion.phase in ("complete", "failed"):
                segment = self._motion_segment
                used = sample["sequence"] - segment["effective_after_sequence"]
                if motion.phase == "complete" and used < STOP_SAMPLES:
                    motion.phase = "braking"
                    evidence.update(phase="braking", completed=False)
                    guard = None
                else:
                    guard = {"reason": "metric_target_reached" if motion.phase == "complete" else "metric_target_failed",
                             "run_task_id": self._run_task_id, "execution_id": snapshot["execution_id"],
                             "generation": snapshot["generation"], "command_request_id": segment["request_id"],
                             "episode_id": sample["episode_id"], "sequence": sample["sequence"],
                             "command": segment["command"], "used_control_steps": used,
                             "max_control_steps": segment["max_control_steps"], "metric_motion": evidence}
            elif (previous_phase != motion.phase or guard is not None or
                  list(motion.command(state)["twist"]) != list(self._motion_segment["command"]["twist"])):
                command = await self._device.on_owner(lambda: backend.set_command(
                    motion.command(state), request_id="metric:" + uuid4().hex))
                self._bind_motion_segment(command, self.MAX_COMMAND_STEPS,
                                          preserve_guard=previous_phase == motion.phase)
                guard = None
            if guard is not None:
                guard["metric_request_id"] = motion.request_id
                guard["command_admission"] = {key: self._motion_segment[key] for key in
                                              ("request_id", "effective_after_sequence", "max_control_steps", "command")}
        if guard is None:
            return
        self._motion_guard = guard
        self._guard_tof_boundary_id = None
        self._last_control["motion_guard"] = guard
        ready = asyncio.Event()
        cleanup = asyncio.create_task(self._complete_motion_pause(ready, gate, self._policy))
        self._motion_cleanup_task = cleanup
        try:
            await gate.pause("planner_pause")
        finally:
            ready.set()

    async def _publish_pausing(self, observation: NativeObservation) -> None:
        if self._require_gate().snapshot()["stop_reason"] in ("budget_exhausted", "policy_stop"):
            return
        await self._publish(observation, self._last_control)

    async def pause(self, arguments: dict[str, Any], *, terminal: bool = False) -> dict[str, Any]:
        async with self._control_lock:
            await self._await_motion_cleanup()
            gate = self._require_gate()
            if "execution_id" in arguments and arguments["execution_id"] != gate.snapshot()["execution_id"]:
                raise ValueError("Stop request belongs to another execution")
            if gate.snapshot()["state"] == "ended":
                return {"status": self._status,
                        "observation": self._observation_wire(self._latest_observation)}
            reason = "user_stop" if terminal else "planner_pause"
            await gate.pause(reason, terminal=terminal)
            await self._await_motion_cleanup()
            if self._policy is not None:
                await self._policy.close()
                self._policy = None
            if self._pump is not None:
                await self._pump
                self._pump = None
            await self._require_device().on_owner(
                self._environment._backend().discard_pending_inference)
            observation = await self._require_device().on_owner(self._environment.observe)
            publication = await self._publish(observation, self._last_control)
            if self._metric_motion is not None and self._metric_motion.phase in ("moving", "braking"):
                self._metric_motion.phase = "interrupted"
                if self._metric_motion.result is not None:
                    self._metric_motion.result.update(phase="interrupted", completed=False, reason=reason)
            return publication

    async def close(self) -> dict[str, Any]:
        try:
            return await super().close()
        finally:
            if self._control_server is not None:
                self._control_server.close()
                await self._control_server.wait_closed()
            if self._policy_server is not None:
                self._policy_server.close()
                await self._policy_server.wait_closed()


async def serve() -> None:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(3, "wb", buffering=0))
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    output_lock = asyncio.Lock()

    async def emit(message: dict[str, Any]) -> None:
        serialized = json.dumps(message, allow_nan=False, separators=(",", ":"))
        if len(serialized) > 32 * 1024 * 1024:
            raise ValueError("Native worker message exceeds the transport bound")
        async with output_lock:
            writer.write((serialized + "\n").encode())
            async with asyncio.timeout(10):
                await writer.drain()

    session = MicroDuckWorkerSession(emit)
    handlers = {
        "initialize": session.initialize,
        "open_task": session.open_task,
        "start": session.start,
        "pause": session.pause,
        "stop": lambda args: session.pause(args, terminal=True),
        "resume": session.resume,
        "capture": session.capture,
        "turn_view": session.turn_view,
        "check": session.check,
        "close_task": lambda _args: session.close_task(),
        "close": lambda _args: session.close(),
    }
    active: set[asyncio.Task[None]] = set()

    async def handle(message: dict[str, Any]) -> None:
        request_id = message["id"]
        try:
            operation = message["op"]
            if operation not in handlers:
                raise ValueError("Unknown native worker operation")
            result = await handlers[operation](require_object(message.get("args", {})))
            await emit({"id": request_id, "result": result})
        except Exception as error:
            await emit({"id": request_id, "error": {"type": type(error).__name__,
                                                   "message": str(error)}})

    while line := await asyncio.to_thread(sys.stdin.readline):
        if len(line) > 32 * 1024 * 1024:
            raise ValueError("Native worker request exceeds the transport bound")
        task = asyncio.create_task(handle(require_object(json.loads(line))))
        active.add(task)
        task.add_done_callback(active.discard)
    await session.transport_disconnected()
    for task in active:
        task.cancel()
    if active:
        await asyncio.gather(*active, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(serve())
