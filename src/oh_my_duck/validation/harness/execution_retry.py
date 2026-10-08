import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

from oh_my_duck.core.paths import project_root


async def run(arguments):
    import torch

    from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
    from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
    from oh_my_duck.validation.harness.control import available_port, control, wait_boundary
    from oh_my_duck.validation.metric.case import require_upright_stop
    from oh_my_duck.validation.metric.policy import verify

    root = project_root()
    configuration = json.loads(arguments.scene_config.read_text())
    if configuration["backend"] != "cpu-mujoco-bam" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Execution retry validation requires CPU MuJoCo/BAM and empty CUDA_VISIBLE_DEVICES")
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    configuration.update(native_task_id="execution-retry", catalog_dir=str(arguments.catalog),
                         policy_revision=OFFICIAL_REVISION, control_port=port, control_secret=secret,
                         seed=arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=False)
    events, samples, calls, publications = [], [], [], {}
    record = {"passed": False, "formal_navigation_acceptance_performed": False,
              "gpu_acceptance_performed": False,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_has_tracked_changes": bool(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy_revision": OFFICIAL_REVISION, "seed": arguments.seed,
              "scene_sha256": hashlib.sha256(arguments.scene_config.read_bytes()).hexdigest(),
              "catalog_manifest_sha256": hashlib.sha256((arguments.catalog / "manifest.json").read_bytes()).hexdigest(),
              "executions": [], "rejections": [], "read_only_checks": []}

    def physical_state():
        backend = session._environment._backend()
        return {"episode_id": backend.episode_id, "sequence": backend._sequence,
                "qpos": backend.data.qpos.tolist(), "qvel": backend.data.qvel.tolist(),
                "ctrl": backend.data.ctrl.tolist(), "physics_time_s": float(backend.data.time),
                "stopped_samples": backend._stopped_samples, "goal_held_ticks": backend._goal_held_ticks}

    async def emit(message):
        events.append(message)
        if message["event"] == "update" and message.get("data", {}).get("observation"):
            observation = session._latest_observation
            control_evidence = message["data"].get("control")
            samples.append({**session._environment._navigation_samples[observation.observation_id],
                            "control": control_evidence})
            if control_evidence is None:
                publications[observation.observation_id] = await session._device.on_owner(physical_state)

    session = MicroDuckWorkerSession(emit)

    async def call(operation, parameters):
        result = await control(port, secret, task_id, operation, parameters)
        calls.append({"operation": operation, "arguments": parameters, "result": result})
        return result

    def save():
        for name, value in (("events", events), ("samples", samples), ("tools", calls), ("result", record)):
            (arguments.output / (name + ".json")).write_text(json.dumps(value, allow_nan=False) + "\n")

    def start_request(attempt):
        return {"native_task_id": "execution-retry", "request": {
            "schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
            "goal_id": "execution-retry", "attempt_id": f"attempt-{attempt}",
            "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
            "required_capabilities": ["policy-navigation"], "success_contract": {
                "id": "execution-retry", "version": "1", "all": [
                    {"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                "source": {"kind": "benchmark", "reference": "execution-retry"}},
            "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
            "owner_assignment_id": "execution-retry", "idempotency_key": uuid4().hex}}

    async def rejected_start(request, expected):
        before = await session._device.on_owner(physical_state)
        segment = deepcopy(session._motion_segment)
        identity = {key: segment[key] for key in ("execution_id", "generation", "request_id",
                                                 "effective_after_sequence", "max_control_steps", "command")}
        try:
            await session.start(request)
        except (ValueError, RuntimeError) as error:
            if expected not in str(error):
                raise
        else:
            raise AssertionError("Native start admitted a request requiring rejection")
        after = await session._device.on_owner(physical_state)
        assert before == after
        assert identity == {key: session._motion_segment[key] for key in identity}
        record["rejections"].append({"reason": expected, "sequence": before["sequence"],
                                      "physical_state_and_command_unchanged": True})

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
            "native_task_id": "execution-retry", "policy_id": "official-microduck-onnx",
            "execution_mode": "policy", "monitor_every_actions": 1,
            "schema_path": str(arguments.edh_source / "harness/contracts/schema/physical.schema.json")})
        from OpenGL import GL

        record["renderer"] = await session._device.on_owner(lambda: GL.glGetString(GL.GL_RENDERER).decode())
        if not any(name in record["renderer"].lower() for name in ("llvmpipe", "softpipe")):
            raise RuntimeError("Execution retry validation requires a Mesa software renderer")
        await session.open_task({"native_task_id": "execution-retry", "run_task_id": task_id,
                                 "catalog_task_id": "execution-retry"})
        prior_request_ids, execution_ids = set(), set()
        for attempt, count in ((1, 75), (2, 75), (3, 50)):
            explicit = None
            if attempt == 3:
                explicit = await call("set_command", {"command": {"twist": [0, 0, 0]},
                                                       "max_control_steps": count})
            before = await session._device.on_owner(physical_state)
            prepared_id = session._motion_segment["request_id"]
            request = start_request(attempt)
            publication = await session.start(request)
            observed_id = session._latest_observation.observation_id
            assert publications[observed_id] == before
            admitted = session._gate.snapshot()
            assert admitted["state"] == "running" and admitted["generation"] == 0
            await wait_boundary(session, timeout_s=None)
            progress = await call("progress", {})
            require_upright_stop(progress)
            segment = session._motion_segment
            snapshot = session._gate.snapshot()
            execution_id = snapshot["execution_id"]
            assert execution_id not in execution_ids
            execution_ids.add(execution_id)
            assert segment["execution_id"] == execution_id == admitted["execution_id"]
            assert segment["generation"] == admitted["generation"]
            assert segment["effective_after_sequence"] == before["sequence"]
            assert segment["max_control_steps"] == count
            assert progress["sequence"] - before["sequence"] == count
            assert session._device.executed_actions == count and session._device.raw_sim_steps == count * 4
            assert segment["request_id"] not in prior_request_ids
            if attempt == 2:
                assert segment["request_id"] != prepared_id
            else:
                assert segment["request_id"] == prepared_id
            prior_request_ids.add(segment["request_id"])
            if explicit is not None:
                assert segment["request_id"] == explicit["request_id"]
            await rejected_start(request, "already admitted")
            await rejected_start(start_request(attempt + 10), "confirmed terminal boundary")
            before_reads = await session._device.on_owner(physical_state)
            observed = await call("observe", {})
            waited = await call("wait_for_motion", {})
            reread = await call("progress", {})
            after_reads = await session._device.on_owner(physical_state)
            assert before_reads == after_reads
            goal = progress["goal_check"]
            assert observed["goal_check"] == waited["goal_check"] == reread["goal_check"] == goal
            assert goal["sequence"] == progress["sequence"] and goal["episode_id"] == progress["episode_id"]
            assert not goal["complete"] and not goal["checks"]["goal_reached"]["satisfied"]
            record["read_only_checks"].append({"sequence": progress["sequence"], "goal_check": goal,
                                               "physics_and_hold_counters_unchanged": True})
            finished = await call("finish_policy", {key: progress["execution"][key]
                                                     for key in ("execution_id", "generation", "boundary_id")})
            terminal = session._gate.snapshot()
            assert terminal["state"] == "ended" and terminal["device_confirmed"]
            assert terminal["stop_reason"] == "policy_stop"
            record["executions"].append({"attempt": attempt, "execution_id": execution_id,
                "command_request_id": segment["request_id"], "initial_physical_state": before,
                "physical_state_preserved_at_start": True, "control_steps": count,
                "raw_sim_steps": session._device.raw_sim_steps, "explicit_command_preserved": explicit is not None,
                "admitted_generation": admitted["generation"],
                "final_progress": progress, "finish": finished, "terminal_boundary": terminal})
            print(json.dumps({"attempt": attempt, "sequence": progress["sequence"],
                              "controls": count, "terminal": terminal["stop_reason"]}), flush=True)
        save()
        record["policy_audit"] = verify(arguments.output, arguments.catalog)
        assert record["policy_audit"]["control_steps"] == 200
        record["physical_control_steps"] = 200
        record["raw_sim_steps"] = 800
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        closed = await session.close()
        released = (closed["closed"] and (session._device is None or session._device._closed)
                    and (session._control_server is None or not session._control_server.is_serving())
                    and (session._policy_server is None or not session._policy_server.is_serving()))
        record.update(closed=closed, resources_released=released, cuda_initialized=torch.cuda.is_initialized())
        assert released and not record["cuda_initialized"]
        record["passed"] = error is None
        save()
    print(json.dumps({"passed": record["passed"], "executions": len(record["executions"]),
                      "physical_control_steps": record["physical_control_steps"],
                      "resources_released": record["resources_released"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    arguments = parser.parse_args()
    for name in ("scene_config", "catalog", "edh_source"):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    asyncio.run(run(arguments))
