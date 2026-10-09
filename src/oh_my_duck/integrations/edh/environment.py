from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from physical_harness.environments import (
    NativeCheck, NativeEnvironmentDescription, NativeFrame, NativeObservation, NativeStep,
)
from physical_harness.execution.worker import require_object

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.integrations.edh.scene_configuration import perception_sources


def _wire_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class MicroDuckEnvironment:
    def __init__(self, configuration: Mapping[str, object]) -> None:
        self.configuration = dict(configuration)
        self.perception_sources = perception_sources(self.configuration)
        self.backend: CpuMujocoBamBackend | None = None
        self._description: NativeEnvironmentDescription | None = None
        self._last_sequence = -1
        self._last_control_started_at: float | None = None
        self._observations: dict[str, tuple[str, int]] = {}
        self._navigation_samples: dict[str, dict[str, Any]] = {}
        self._pending_policy_evidence: dict[str, Any] | None = None
        self._last_action_inference: dict[str, Any] | None = None

    def require_perception_source(self, source: str) -> None:
        if source not in self.perception_sources:
            raise ValueError(f"Perception source {source!r} is unavailable; available sources: "
                             f"{', '.join(self.perception_sources)}")

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
            "body_twist_world": tuple(measured["body_twist_world"]),
        }
        inference = self._last_action_inference
        if inference is not None and (inference["episode_id"], inference["result_sequence"]) == (state["episode_id"], sequence):
            self._navigation_samples[observation_id]["policy_inference"] = inference
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
        registry = (Path(str(configuration["policy_registry"])).resolve(strict=True)
                    if configuration.get("policy_registry") is not None else None)
        backend_name = configuration.get("backend", "cpu-mujoco-bam")
        if backend_name == "cpu-mujoco-bam":
            self.backend = CpuMujocoBamBackend(robot_id="microduck", catalog_dir=catalog,
                                             policy_registry=registry)
        elif backend_name == "isaac-newton":
            from oh_my_duck.robotics.runtime_startup import RuntimeStartup

            startup = RuntimeStartup(str(configuration["scene_id"]), str(configuration.get("robot_model", "allcollisions")))
            startup.mark("backend_import")
            from oh_my_duck.robotics.backends.isaac_official import IsaacNewtonBamBackend

            self.backend = IsaacNewtonBamBackend(
                robot_id="microduck", catalog_dir=catalog, policy_registry=registry,
                scene_path=Path(str(configuration["usd_path"])).resolve(strict=True),
                device=str(configuration.get("device", "cuda:0")),
                scene_id=str(configuration["scene_id"]),
                robot_model=str(configuration.get("robot_model", "allcollisions")),
                observer_renderer=str(configuration.get("observer_renderer", "newton_warp")),
                startup=startup,
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
                            "policy_revision": self.backend.catalog.revision,
                            "perception_sources": list(self.perception_sources),
                            "seed": configuration["seed"], "spawn_pose": configuration.get("spawn_pose")},
        )
        return self._observation(state)

    def bind_task(self, task_id: str) -> None:
        if task_id != self.configuration["native_task_id"]:
            raise ValueError("MicroDuck 保留环境的任务编号不同")

    def begin_task_goal(self, task_id: str) -> None:
        self.bind_task(task_id)
        self._backend().begin_task_goal()

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
        inference = self._pending_policy_evidence
        if (inference is None or (inference["episode_id"], inference["sequence"]) != (backend.episode_id, backend._sequence)
                or list(action) != inference["action"]):
            raise RuntimeError("Admitted action lacks its matching policy inference evidence")
        result = backend.apply_policy_action(list(action), request_id=uuid4().hex,
                                             expected_sequence=backend._sequence,
                                             should_stop=should_stop)
        raw_steps = result["raw_sim_steps"]
        if type(raw_steps) is not int or not 0 <= raw_steps <= 4:
            raise RuntimeError("MicroDuck returned an invalid physical step count")
        if raw_steps:
            self._last_action_inference = {**inference, "result_sequence": result["sequence"],
                                           "raw_sim_steps": raw_steps,
                                           "action_request_id": result["request_id"]}
        self._pending_policy_evidence = None
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
