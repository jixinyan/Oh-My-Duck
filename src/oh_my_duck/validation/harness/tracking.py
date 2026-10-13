import argparse
import asyncio
import base64
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from uuid import uuid4

from oh_my_duck.core.paths import project_root


def _save(path: Path, value) -> None:
    with path.open("x") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")


async def run_case(args, source: str, output: Path) -> dict:
    import numpy as np
    import torch

    from oh_my_duck.integrations.edh.session import MicroDuckWorkerSession
    from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION
    from oh_my_duck.validation.harness.control import available_port, control, expect_control_error, wait_boundary

    output.mkdir()
    (output / "frames").mkdir()
    port, secret, task_id = available_port(), secrets.token_hex(32), uuid4().hex
    prompt = "ball" if source == "simulator_ground_truth" else "objects"
    configuration = {"backend": "cpu-mujoco-bam", "native_task_id": "tracking-tools",
        "catalog_dir": str(args.catalog.resolve(strict=True)), "seed": args.seed,
        "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
        "spawn_pose": {"x_m": -3.65, "y_m": 0, "yaw_rad": 0} if source == "simulator_ground_truth"
                      else {"x_m": -1.6, "y_m": -0.5, "yaw_rad": math.pi},
        "task_instruction": "使用当前相机测量更新目标身份并查询跟踪状态",
        "policy_revision": OFFICIAL_REVISION, "control_port": port, "control_secret": secret,
        "perception_endpoint": args.endpoint}
    events, samples, tools, captures, frames = [], [], [], [], []
    phase = "warmup"
    record = {"passed": False, "source": source, "prompt": prompt,
              "configuration": {key: value for key, value in configuration.items() if key != "control_secret"}}

    async def emit(message):
        events.append(message)
        data = message.get("data", {})
        if message["event"] == "update" and data.get("observation"):
            observation = session._latest_observation
            sample = session._environment._navigation_samples[observation.observation_id]
            samples.append({**sample, "phase": phase, "control": data.get("control"),
                            "body_twist": list(observation.state["body_twist"])})
        elif message["event"] == "frame":
            name = f"frames/{len(frames):06d}.png"
            image = data["observation"]["images"]["observer_follow"]
            (output / name).write_bytes(base64.b64decode(image, validate=True))
            frames.append({"file": name, "simulation_time_s": data["simulation_time_s"], "phase": phase})

    session = MicroDuckWorkerSession(emit)

    async def call(operation, arguments):
        result = await control(port, secret, task_id, operation, arguments)
        tools.append({"operation": operation, "arguments": arguments, "result": deepcopy(result), "phase": phase})
        return result

    async def advance(count, yaw=0.0):
        await call("set_command", {"command": {"twist": [0.0] * 3, "head": [0.0, 0.0, yaw, 0.0]},
                                   "max_control_steps": count})
        status = session._status
        await session.resume({"owner_id": "planner", "execution_id": status["execution_id"],
                              "boundary_id": status["boundary_event_id"], "state_version": status["state_version"]})
        await wait_boundary(session)

    async def inspect():
        result = await call("inspect_scene", {"prompt": prompt, "source": source})
        backend = session._environment._backend()

        def physical_geometry():
            frame, segmentation = backend._perception_capture()
            return frame, segmentation, [backend.model.geom(index).name for index in range(backend.model.ngeom)],

        frame, segmentation, names = await session._device.on_owner(physical_geometry)
        index = len(captures)
        geometry = f"capture-{index:03d}-geometry.npz"
        np.savez_compressed(output / geometry, segmentation=segmentation)
        capture = {"phase": phase, "perception": deepcopy(result), "geometry": geometry,
                   "geom_names": names, "physical_sequence": frame["sequence"]}
        assert result["sequence"] == frame["sequence"] and result["episode_id"] == frame["episode_id"]
        if source == "models":
            matching = []
            for path in args.inspections.glob("*/inspection.json"):
                metadata = json.loads(path.read_text())
                if all(metadata[field] == result[field] for field in ("episode_id", "sequence", "observed_at", "prompt")):
                    matching.append((path.parent, metadata))
            if len(matching) != 1:
                raise AssertionError("Native model call requires its unique original service capture")
            directory, metadata = matching[0]
            for field, name in (("frame", "frame.json"), ("response", "response.json")):
                captured = f"capture-{index:03d}-{name}"
                shutil.copyfile(directory / name, output / captured)
                assert hashlib.sha256((output / captured).read_bytes()).hexdigest() == metadata["files"][name]
                capture[field] = captured
            capture["device"] = metadata["device"]
            original = json.loads((output / capture["frame"]).read_text())
            assert original["rgb_png_base64"] == frame["rgb_png_base64"]
            assert original["points_world_npy_base64"] == frame["points_world_npy_base64"]
        else:
            captured = f"capture-{index:03d}-physical-frame.json"
            _save(output / captured, frame)
            capture["physical_frame"] = captured
        captures.append(capture)
        return result

    try:
        await session.initialize({"provider": "microduck", "scene_configuration": configuration,
            "native_task_id": "tracking-tools", "policy_id": "official-microduck-onnx", "execution_mode": "policy",
            "schema_path": str(args.edh_source.resolve(strict=True) / "harness/contracts/schema/physical.schema.json"),
            "monitor_every_actions": 1, "policy_max_actions_per_inference": 1})
        from OpenGL import GL

        record["renderer"] = await session._device.on_owner(lambda: GL.glGetString(GL.GL_RENDERER).decode())
        assert "llvmpipe" in record["renderer"].lower()
        await session.open_task({"native_task_id": "tracking-tools", "run_task_id": task_id,
                                 "catalog_task_id": "tracking-tools"})
        request = {"schema_version": "physical.subgoal.v1", "task_id": task_id, "team_run_id": task_id,
            "goal_id": "tracking-tools", "attempt_id": "tracking", "instruction": configuration["task_instruction"],
            "entities": {"robot": "microduck"}, "required_capabilities": ["head-rgb"],
            "success_contract": {"id": "tracking-tools", "version": "1",
                "all": [{"check_id": "goal_reached", "check": "native_goal_reached", "args": []}],
                "source": {"kind": "benchmark", "reference": "native-tracking-tools"}},
            "budget": {"max_control_steps": 700, "max_wall_time_s": 600}, "context_refs": [],
            "decision_owner_id": "planner", "owner_assignment_id": "tracking-tools", "idempotency_key": uuid4().hex}
        await session.start({"request": request, "native_task_id": "tracking-tools"})
        await wait_boundary(session)
        phase = "initial"
        initial = await inspect()
        candidates = [target for target in initial["targets"] if target["track_id"] is not None]
        if source == "simulator_ground_truth":
            candidates = [target for target in candidates if target["target_id"] == "ball_0"]
        else:
            candidates = [target for target in candidates if target["label"] == "chair"]
        if not candidates:
            raise AssertionError("Actual scene and model require at least one measured target")
        target = candidates[0]
        identifier = target["track_id"]
        first = await call("read_track", {"track_id": identifier})
        assert first["status"] == ("current" if source == "simulator_ground_truth" else "tentative")
        repeated = await inspect()
        same = next(item for item in repeated["targets"] if item["track_id"] == identifier)
        assert same["tracking"]["observation_count"] == 1
        phase = "updated"
        await advance(25)
        assert (await call("read_track", {"track_id": identifier}))["status"] == "stale"
        updated = await inspect()
        same = next(item for item in updated["targets"] if item["track_id"] == identifier)
        assert same["tracking"]["observation_count"] == 2
        assert (await call("read_track", {"track_id": identifier}))["status"] == "current"
        for arguments, message in (({"track_id": "unknown"}, "Unknown"),
                ({"track_id": identifier, "max_age_ms": True}, "max_age_ms"),
                ({"track_id": identifier, "max_age_ms": 0}, "max_age_ms"),
                ({"track_id": identifier, "extra": 1}, "unsupported")):
            await expect_control_error(port, secret, task_id, "read_track", arguments, message)
        if source == "simulator_ground_truth":
            phase = "moving_instance"
            backend = session._environment._backend()

            def push_ball():
                identifier = backend.model.body("ball_0").id
                start = backend.data.xpos[identifier].copy()
                backend.data.xfrc_applied[identifier, 0] = 0.003
                return identifier, start

            body_id, start = await session._device.on_owner(push_ball)
            await advance(75)

            def stop_force():
                backend.data.xfrc_applied[body_id] = 0
                return backend.data.xpos[body_id].copy()

            finish = await session._device.on_owner(stop_force)
            record["object_motion"] = {"body": "ball_0", "force_world_n": [0.003, 0.0, 0.0],
                "start_position_m": start.tolist(), "final_position_m": finish.tolist(),
                "translation_m": float(np.linalg.norm(finish - start))}
            assert record["object_motion"]["translation_m"] > 0.02
            moved = await inspect()
            assert next(item for item in moved["targets"] if item["target_id"] == "ball_0")["track_id"] == identifier
            assert (await call("read_track", {"track_id": identifier}))["status"] == "current"
        else:
            phase = "look_away"
            await advance(75, yaw=1.4)
            await inspect()
            away = await call("read_track", {"track_id": identifier})
            record["look_away_status"] = away["status"]
            phase = "look_return"
            await advance(50)
            await inspect()
            record["look_return_status"] = (await call("read_track", {"track_id": identifier}))["status"]
            phase = "reconfirmed"
            await advance(25)
            await inspect()
            record["reconfirmed_status"] = (await call("read_track", {"track_id": identifier}))["status"]
        final = await call("progress", {})
        record.update(track_id=identifier, control_steps=final["sequence"], raw_sim_steps=session._status["raw_sim_steps"],
                      final_execution=final["execution"], invalid_queries_rejected=4)
        await session.pause({"execution_id": session._gate.snapshot()["execution_id"]}, terminal=True)
        sequence = (await call("progress", {}))["sequence"]
        await asyncio.sleep(0.2)
        assert (await call("progress", {}))["sequence"] == sequence
        task_id = uuid4().hex
        await session.open_task({"native_task_id": "tracking-tools", "run_task_id": task_id,
                                 "catalog_task_id": "tracking-tools"})
        await expect_control_error(port, secret, task_id, "read_track", {"track_id": identifier}, "current physical episode")
        record.update(passed=True, new_task_clears_tracks=True, cuda_initialized=torch.cuda.is_initialized())
        assert not record["cuda_initialized"]
    finally:
        await session.close()
        record["resources_closed"] = not session._control_server.is_serving() and not session._policy_server.is_serving()
        for name, value in (("result", record), ("events", events), ("samples", samples), ("tools", tools),
                            ("captures", captures), ("frames", frames)):
            _save(output / (name + ".json"), value)
    return record


async def run(args):
    root = project_root()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU tracking validation requires empty CUDA_VISIBLE_DEVICES")
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / "provenance.json", {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_has_tracked_changes": bool(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    cases = [await run_case(args, source, args.output / source)
             for source in ("simulator_ground_truth", "models")]
    _save(args.output / "result.json", {"passed": True, "cases": cases, "gpu_acceptance_performed": False,
        "formal_navigation_acceptance_performed": False, "autonomous_following_acceptance_performed": False})
    print(json.dumps({"passed": True, "cases": len(cases), "control_steps": sum(case["control_steps"] for case in cases),
                      "resources_closed": all(case["resources_closed"] for case in cases)}))


def main():
    parser = argparse.ArgumentParser(description="通过官方 CPU policy 与原生工具验证实际目标跟踪")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--inspections", type=Path, required=True,
                        help="Actual service --record-inspections output directory")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
