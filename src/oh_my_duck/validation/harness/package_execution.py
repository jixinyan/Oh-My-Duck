import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

from oh_my_duck.core.paths import project_root


async def run(args):
    import torch

    from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession
    from oh_my_duck.validation.harness.control import available_port, control, expect_control_error
    from oh_my_duck.validation.metric.policy import verify

    root = project_root()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    port, secret, task_id = available_port(), secrets.token_hex(32), "packages-" + uuid4().hex
    configuration = json.loads(args.scene_config.read_text())
    if configuration["backend"] != "cpu-mujoco-bam":
        raise ValueError("Policy package execution requires the official CPU apartment")
    configuration.update(native_task_id="policy-packages", catalog_dir=str(args.catalog),
                         policy_registry=str(args.registry), control_port=port, control_secret=secret,
                         policy_revision="registered-schema2", seed=args.seed)
    record = {"passed": False, "scope": "Registered policies through native tools and physical control",
              "perpetual_policy": args.perpetual_policy, "episodic_policy": args.episodic_policy,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_has_tracked_changes": bool(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
              "scene_sha256": hashlib.sha256(args.scene_config.read_bytes()).hexdigest(),
              "behavior_acceptance_performed": False, "model_task_acceptance_performed": False,
              "gpu_acceptance_performed": False}
    events, samples, tools, frames = [], [], [], []
    phase = "warmup"

    async def emit(message):
        events.append(message)
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            sample = session._environment._navigation_samples[observation.observation_id]
            samples.append({**sample, "phase": phase, "control": data.get("control")})
        elif message["event"] == "frame":
            relative = f"frames/{len(frames):06d}.png"
            (args.output / relative).write_bytes(base64.b64decode(
                data["observation"]["images"]["observer_follow"], validate=True))
            frames.append({"file": relative, "simulation_time_s": data["simulation_time_s"], "phase": phase})

    session = MicroDuckWorkerSession(emit)

    def save():
        for name, value in (("result", record), ("events", events), ("samples", samples),
                            ("tools", tools), ("frames", frames)):
            (args.output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    async def call(operation, arguments):
        result = await control(port, secret, task_id, operation, arguments)
        tools.append({"phase": phase, "operation": operation, "arguments": arguments, "result": result})
        return result

    async def wait():
        async with asyncio.timeout(180):
            await session._pump
            await session._await_motion_cleanup()
        snapshot = session._gate.snapshot()
        if snapshot["state"] not in ("paused", "ended") or not snapshot["device_confirmed"]:
            raise AssertionError("Policy package execution lacks a confirmed native boundary")
        return snapshot

    async def start(attempt):
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
                   "goal_id": "policy-packages", "attempt_id": attempt,
                   "instruction": configuration["task_instruction"], "entities": {"robot": "microduck"},
                   "required_capabilities": ["policy-navigation"],
                   "success_contract": {"id": "policy-packages", "version": "1",
                       "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                       "source": {"kind": "benchmark", "reference": "registered-package-control"}},
                   "budget": configuration["budget"], "context_refs": [], "decision_owner_id": "planner",
                   "owner_assignment_id": "package-validation", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "policy-packages",
                             "observation_ttl_s": 30, "device_timeout_s": 120, "policy_timeout_s": 30})
        return await wait()

    async def segment(twist, count):
        await call("set_command", {"command": {"twist": twist}, "max_control_steps": count})
        prior = session._status
        await session.resume({"owner_id": "planner", "execution_id": prior["execution_id"],
                              "boundary_id": prior["boundary_event_id"], "state_version": prior["state_version"]})
        boundary = await wait()
        await call("observe", {})
        save()
        return boundary

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
            "native_task_id": "policy-packages", "policy_id": "registered-schema2-onnx",
            "execution_mode": "policy", "schema_path": str(args.edh_source / "harness/contracts/schema/physical.schema.json"),
            "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        from OpenGL import GL

        record["renderer"] = await session._device.on_owner(lambda: GL.glGetString(GL.GL_RENDERER).decode())
        if not any(name in record["renderer"].lower() for name in ("llvmpipe", "softpipe")):
            raise RuntimeError("CPU package execution requires Mesa software rendering")
        await session.open_task({"native_task_id": "policy-packages", "run_task_id": task_id,
                                 "catalog_task_id": "policy-packages"})
        record["catalogue"] = await call("catalog", {})
        descriptions = {policy["name"]: policy for policy in record["catalogue"]["policies"]}
        for name, kind in ((args.perpetual_policy, "perpetual"), (args.episodic_policy, "episodic")):
            policy = descriptions[name]
            if policy.get("source") != "registered_schema2_package" or policy["kind"] != kind:
                raise ValueError("Execution requires actual registered perpetual and episodic packages")
        if "twist" not in descriptions[args.perpetual_policy]["command_channels"]:
            raise ValueError("Perpetual package must declare twist commands")
        await start("packages")
        phase = "perpetual"
        record["metric_preparation"] = await call("walk", {"distance_m": 0.2})
        if record["metric_preparation"]["policy_name"] != args.perpetual_policy:
            raise AssertionError("Metric tool did not select the registered locomotion policy")
        await segment([0.1, 0.0, 0.0], 50)
        phase = "perpetual_stop"
        await segment([0.0] * 3, 100)
        await call("select_policy", {"policy_name": args.episodic_policy})
        phase = "episodic"
        await expect_control_error(port, secret, task_id, "set_command",
            {"command": {"twist": [0.1, 0.0, 0.0]}}, "does not accept a velocity command")
        count = round(descriptions[args.episodic_policy]["duration_s"] * 50)
        if abs(count / 50 - descriptions[args.episodic_policy]["duration_s"]) > 1e-9:
            raise ValueError("Package duration must match complete 50 Hz control periods")
        for offset in range(0, count, 100):
            boundary = await segment([0.0] * 3, max(5, min(100, count - offset)))
        if boundary["state"] != "ended" or boundary["stop_reason"] != "episode_terminated":
            raise AssertionError("Complete package duration did not produce native episode termination")
        before = await call("progress", {})
        record["transition"] = await call("transition_policy", {"policy_name": "alpha_stand"})
        after = await call("progress", {})
        for key in ("episode_id", "sequence", "simulation_time_s", "body_position_m", "body_twist"):
            if before[key] != after[key]:
                raise AssertionError(f"Package transition changed physical state: {key}")
        phase = "recovery"
        await call("set_command", {"command": {"twist": [0.0] * 3}, "max_control_steps": 100})
        record["recovery_boundary"] = await start("recovery")
        final = await call("progress", {})
        record["final_progress"] = final
        record["termination"] = await call("finish_policy", {
            key: final["execution"][key] for key in ("execution_id", "generation", "boundary_id")})
        record["physical_control_steps"] = final["sequence"]
        if final["sequence"] != 75 + 50 + 100 + count + 100:
            raise AssertionError("Recorded package controls differ from declared phases")
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        record["closed"] = await session.close()
        record["resources_released"] = (record["closed"]["closed"]
            and (session._device is None or session._device._closed)
            and (session._control_server is None or not session._control_server.is_serving())
            and (session._policy_server is None or not session._policy_server.is_serving()))
        record["cuda_initialized"] = torch.cuda.is_initialized()
        save()
        if not record["resources_released"] or record["cuda_initialized"]:
            raise AssertionError("CPU package execution did not release resources without CUDA")
    record["policy_audit"] = verify(args.output, args.catalog, args.registry)
    record["passed"] = True
    save()
    return record


def main():
    parser = argparse.ArgumentParser()
    for name in ("scene-config", "catalog", "registry", "edh-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--perpetual-policy", required=True)
    parser.add_argument("--episodic-policy", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU package execution requires CUDA_VISIBLE_DEVICES to be empty")
    for name in ("scene_config", "catalog", "registry", "edh_source"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    result = asyncio.run(run(args))
    print(json.dumps({"passed": result["passed"], "controls": result["physical_control_steps"],
                      "policy_audit": result["policy_audit"]}, allow_nan=False))


if __name__ == "__main__":
    main()
