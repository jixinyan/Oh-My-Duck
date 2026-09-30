from __future__ import annotations

import asyncio
import math
import time
from uuid import uuid4

from oh_my_duck.core.contracts.tasks import TaskHandle, TaskResult, TaskStatus
from oh_my_duck.robotics.backends.simulation import CONTROL_DT, MotionBusyError, SimulationBackend, _finite_velocity


class SimulationSkillRunner:
    def __init__(self, backend: SimulationBackend):
        self.backend = backend
        self.backend.bind_runner(self)
        self._lock = asyncio.Lock()
        self._active: TaskHandle | None = None
        self._results: dict[str, TaskResult] = {}
        self._requests: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
        self._cancellations: dict[str, asyncio.Event] = {}
        self._jobs: dict[str, asyncio.Task] = {}
        self._manual_stop = False

    async def start(self, skill_id: str, parameters: dict, *, request_id: str) -> TaskHandle:
        if skill_id != "move_for":
            raise ValueError(f"Unsupported skill: {skill_id}")
        if set(parameters) != {"vx_m_s", "vy_m_s", "yaw_rad_s", "duration_s"}:
            raise ValueError("move_for requires vx_m_s, vy_m_s, yaw_rad_s and duration_s")
        velocity = _finite_velocity(parameters["vx_m_s"], parameters["vy_m_s"], parameters["yaw_rad_s"])
        duration = parameters["duration_s"]
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or not CONTROL_DT <= duration <= 60:
            raise ValueError("duration_s must be finite and within [0.02, 60]")
        expected_ticks = round(duration / CONTROL_DT)
        if not math.isclose(duration, expected_ticks * CONTROL_DT, abs_tol=1e-8):
            raise ValueError("duration_s must contain a whole number of 50 Hz control ticks")
        if not request_id:
            raise ValueError("request_id is required")
        specification = (skill_id, (*velocity, float(duration)))
        async with self._lock:
            if request_id in self._requests:
                if self._requests[request_id] != specification:
                    raise ValueError("request_id was already used with different skill parameters")
                return next(result.handle for result in self._results.values() if result.handle.request_id == request_id)
            if self._active is not None or self._manual_stop:
                raise MotionBusyError(f"Robot is busy with task {self._active.task_id if self._active else 'stop_motion'}")
            handle = TaskHandle(uuid4().hex, request_id, self.backend.robot_id)
            self._requests[request_id] = specification
            self._results[handle.task_id] = TaskResult(handle, TaskStatus.ACCEPTED)
            cancellation = asyncio.Event()
            self._cancellations[handle.task_id] = cancellation
            self._active = handle
            self.backend._active_task_id = handle.task_id
            self._jobs[handle.task_id] = asyncio.create_task(self._execute(handle, velocity, float(duration), cancellation))
            return handle

    async def status(self, handle: TaskHandle) -> TaskResult:
        result = self._results.get(handle.task_id)
        if result is None or result.handle != handle:
            raise ValueError("Unknown task handle")
        return result

    def handle(self, task_id: str) -> TaskHandle:
        if task_id not in self._results:
            raise ValueError("Unknown task_id in this execution session")
        return self._results[task_id].handle

    async def cancel(self, handle: TaskHandle, *, reason: str) -> TaskResult:
        async with self._lock:
            result = await self.status(handle)
            if self._active != handle:
                return result
            self._results[handle.task_id] = TaskResult(handle, TaskStatus.CANCELLING, result.evidence_refs, reason)
            self._cancellations[handle.task_id].set()
            job = self._jobs[handle.task_id]
        await job
        return await self.status(handle)

    async def stop_active(self, *, request_id: str) -> TaskResult | object:
        async with self._lock:
            handle = self._active
            if handle is None:
                if self._manual_stop:
                    raise MotionBusyError("Motion stop confirmation is in progress")
                self._manual_stop = True
        if handle is not None:
            return await self.cancel(handle, reason=f"stop_motion:{request_id}")
        try:
            return await self.backend.stop_motion(request_id=request_id)
        finally:
            async with self._lock:
                self._manual_stop = False

    async def _execute(self, handle: TaskHandle, velocity: tuple[float, float, float],
                       duration_s: float, cancellation: asyncio.Event) -> None:
        self._results[handle.task_id] = TaskResult(handle, TaskStatus.RUNNING)
        started = time.monotonic()
        first_sequence = self.backend._sequence
        last_sequence = first_sequence
        command_updates = 0
        applied_ticks = 0
        first_position = None
        last_position = None
        measured_twist = None
        expected_ticks = round(duration_s / CONTROL_DT)
        failure: str | None = None
        try:
            while applied_ticks < expected_ticks and not cancellation.is_set():
                await self.backend.command_velocity(*velocity, request_id=f"{handle.task_id}:{command_updates}", ttl_ms=100,
                    owner_task_id=handle.task_id)
                command_updates += 1
                await asyncio.sleep(CONTROL_DT / 2)
                current = await self.backend.state()
                if current.sequence == last_sequence:
                    continue
                last_sequence = current.sequence
                values = current.measurements
                if first_position is None:
                    first_position = values["body_position_m"]
                last_position = values["body_position_m"]
                measured_twist = values["body_twist"]
                if tuple(values["command"]) == velocity:
                    applied_ticks += 1
                if current.measurements["fallen"] or not current.measurements["valid"]:
                    failure = "Robot fell or simulation state became invalid"
                    break
            stop = await self.backend.stop_motion(request_id=f"{handle.task_id}:stop")
            evidence = (f"simulation:{self.backend.episode_id}:{first_sequence}", *stop.evidence_refs)
            if not stop.confirmed_stopped:
                failure = stop.reason
            if applied_ticks < expected_ticks and not cancellation.is_set() and failure is None:
                failure = "Requested command was not applied for the required control ticks"
            status = TaskStatus.FAILED if failure else TaskStatus.CANCELLED if cancellation.is_set() else TaskStatus.SUCCEEDED
            details = {"clock_domain": "simulation", "requested_duration_s": duration_s,
                "applied_control_ticks": applied_ticks, "expected_control_ticks": expected_ticks,
                "wall_elapsed_s": time.monotonic() - started, "start_position_m": first_position,
                "end_position_m": last_position, "last_measured_twist": measured_twist,
                "confirmed_stopped": stop.confirmed_stopped, "policy_sha256": self.backend.policy_sha256,
                "episode_id": self.backend.episode_id}
            self._results[handle.task_id] = TaskResult(handle, status, tuple(evidence), failure, details)
        except Exception as error:
            self._results[handle.task_id] = TaskResult(handle, TaskStatus.FAILED,
                (f"simulation:{self.backend.episode_id}:{self.backend._sequence}",), str(error))
            raise
        finally:
            async with self._lock:
                if self._active == handle:
                    self._active = None
                    self.backend._active_task_id = None
