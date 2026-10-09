import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import soundfile as sf

from oh_my_duck.voice.audio import _source_path


def verify(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    recorded = json.loads((directory / "task/result.json").read_text())
    tasks = recorded["tasks"]
    if len(tasks) < 2:
        raise ValueError("连续语音验收需要至少两个已完成任务")
    session = json.loads((directory / "task/native/session-closed.json").read_text())
    assert session["state"] == "closed" and session["resources"] == "released"
    latest = {}
    with (directory / "data/records.jsonl").open() as journal:
        for line in journal:
            entry = json.loads(line)
            latest[entry["key"]] = entry["record"]
    reports = []
    previous_run = previous_progress = previous_scope = None
    for index, task in enumerate(tasks, 1):
        replay = directory / f"replay-{index}"
        run = json.loads((replay / "source/run.json").read_text())
        events = json.loads((replay / "source/events.json").read_text())
        assert run["id"] == task["submission"]["runId"] and run["state"] == "succeeded"
        assert run["instruction"] == task["transcription"]["text"]
        assert run["userSessionId"] == session["id"] and run["source"] == "simulation"
        contexts = run["taskContext"]
        if previous_run is None:
            assert contexts == [] and task["context_run_ids"] == []
        else:
            assert len(contexts) == 1 and task["context_run_ids"] == [previous_run["id"]]
            context = contexts[0]
            assert context["runId"] == previous_run["id"] and context["userSessionId"] == session["id"]
            assert context["instruction"] == previous_run["instruction"]
            assert context["outcome"] == "succeeded" and context["source"] == "simulation"
            assert context["finalVerification"]["status"] == "passed"
            assert context["finalVerification"]["verdictId"] in [item["verdict_id"] for item in previous_run["verdicts"]]
            native = latest["run:" + run["id"]]["value"]
            planner = native["assignments"][native["decisionAssignmentId"]]
            assert previous_run["id"] in planner["brief"]["history_summary"]
            assert context["finalVerification"]["verdictId"] in planner["brief"]["history_summary"]
            assert {"skills.search", "skills.load"} <= set(planner["tools"])
        assert latest["run-user-session:" + run["id"]]["value"]["sessionId"] == session["id"]
        assert run["executions"] and all(item["state"] == "ended" and item["device_confirmed"] for item in run["executions"])
        execution = run["executions"][-1]
        assert execution["control_steps"] >= 5 and execution["raw_sim_steps"] == 4 * execution["control_steps"]
        assert execution["stop_reason"] == "policy_stop"
        tools = [item["detail"] for item in events if item["type"] == "tool.completed"]
        progress = [item["result"] for item in tools if item["tool"] in (
            "microduck.observe", "microduck.task_progress", "microduck.wait_for_motion")]
        assert progress and progress[-1]["stopped_samples"] >= 5 and not progress[-1]["fallen"]
        start, finish = progress[0]["task_start"], progress[-1]
        assert start["run_task_id"] == run["id"]
        start_evidence = start["goal_check"]["checks"]["goal_reached"]["evidence"]
        final_evidence = finish["goal_check"]["checks"]["goal_reached"]["evidence"]
        scope = final_evidence["goal_scope_id"]
        assert all(item["task_start"] == start for item in progress)
        assert scope == start_evidence["goal_scope_id"] and scope != previous_scope
        assert start_evidence["held_ticks"] == 0 and not start["goal_check"]["complete"]
        assert final_evidence["goal_bound_sequence"] == start["sequence"]
        assert finish["sequence"] - start["sequence"] >= final_evidence["held_ticks"] >= 5
        assert final_evidence["required_hold_ticks"] == 5 and finish["goal_check"]["complete"]
        assert final_evidence["height_m"] >= final_evidence["upright_minimum_height_m"] == 0.09
        assert final_evidence["tilt_rad"] <= final_evidence["upright_maximum_tilt_rad"]
        if previous_progress is not None:
            assert previous_progress["episode_id"] == start["episode_id"]
            assert previous_progress["sequence"] == start["sequence"]
            assert math.dist(previous_progress["body_position_m"], start["body_position_m"]) < 1e-6
        checked = [item["detail"] for item in events if item["type"] == "verification.checked"]
        completed = [item["detail"]["result"] for item in events if item["type"] == "verification.completed"]
        assert len(checked) == len(completed) == 1
        verdict = completed[0]
        assert verdict["status"] == "passed" and verdict["execution_id"] == execution["execution_id"]
        assert verdict["boundary_event_id"] == execution["boundary_event_id"]
        assert checked[0]["facts"][0]["value"] is verdict["checks"][0]["value"] is True
        assert {"verification__check", "verification__submit", "tasks__finish"} <= {
            item["detail"]["data"]["name"] for item in events if item["type"] == "dsh.tool-call"}
        frames = json.loads((replay / "frames.json").read_text())
        assert frames
        for frame in frames:
            path = (replay / frame["file"]).resolve(strict=True)
            assert path.is_relative_to(replay)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["image"]["attachmentId"].removeprefix("sha256:")
            with Image.open(path) as image:
                assert image.format == "PNG" and image.size == (frame["image"]["width"], frame["image"]["height"])
                image.verify()
        speech = task["speech"]
        assert (speech["persona_id"], speech["voice_id"], speech["profile_revision"]) == (
            tasks[0]["speech"]["persona_id"], tasks[0]["speech"]["voice_id"], tasks[0]["speech"]["profile_revision"])
        audio_path = _source_path(speech["audio"]["uri"])
        assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == speech["audio"]["sha256"]
        with sf.SoundFile(audio_path) as audio:
            assert audio.subtype == "PCM_16" and audio.samplerate == 24000 and audio.channels == 1
            samples = audio.read(dtype="float32")
        assert np.isfinite(samples).all() and np.any(samples != 0)
        search_calls = sum(item["tool"] == "skills.search" for item in tools)
        if previous_run is not None:
            assert search_calls >= 1
        reports.append({"run_id": run["id"], "events": len(events), "goal_scope_id": scope,
                        "policy_control_steps": execution["control_steps"], "raw_sim_steps": execution["raw_sim_steps"],
                        "images_verified": len(frames), "verdict_id": verdict["verdict_id"],
                        "skill_search_calls": search_calls, "stopped_samples": finish["stopped_samples"]})
        previous_run, previous_progress, previous_scope = run, finish, scope
    return {"passed": True, "scope": "连续录音任务、原生历史引用、独立目标检查与已确认音色",
            "tasks": reports, "session_id": session["id"], "session_resources_released": True,
            "gpu_acceptance_performed": False,
            "journal_sha256": hashlib.sha256((directory / "data/records.jsonl").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
