import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

import httpx

from validate_voice_interaction import SessionProcess


async def run(args):
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    process = await asyncio.create_subprocess_exec(sys.executable, str(root / "omd.py"), "voice-session",
        "--persona", args.persona, "--robot-id", "voice-native-interrupt", "--domain", "simulation",
        "--input-device", args.input_device, "--output-device", args.output_device,
        "--asr-url", args.asr_url, "--tts-url", args.tts_url, "--harness-url", args.harness_url,
        "--harness-profile", args.profile, "--harness-scenario", args.scenario,
        "--data-dir", str(args.output / "session"), "--log", str(args.output / "events.jsonl"),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")})
    session = SessionProcess(process)
    try:
        await session.next("voice.session.started")
        await session.send("voice-command", "execute_recording", audio=str(args.audio.resolve(strict=True)),
            language_hint="Chinese")
        transcription = await session.next("voice.transcription.completed")
        submitted = await session.next("voice.task.started")
        run_id = submitted["payload"]["runId"]
        async with httpx.AsyncClient(base_url=args.harness_url, trust_env=False, timeout=30) as client:
            async with asyncio.timeout(600):
                while True:
                    response = await client.get(f"/api/runs/{run_id}?events=none")
                    response.raise_for_status()
                    before = response.json()
                    if any(item["state"] == "running" and item["control_steps"] > 75 for item in before["executions"]):
                        break
                    if before["state"] in {"succeeded", "failed", "cancelled", "interrupted", "unknown"}:
                        raise AssertionError("Task ended before an active-motion interruption")
                    if before["clarification"] is not None:
                        raise AssertionError("Task requested clarification before active-motion interruption")
                    await asyncio.sleep(0.1)
            await session.send("operator-stop", "stop")
            stopped = await session.next("voice.stopped")
            acknowledgement = stopped["payload"]["robot"]
            if acknowledgement["run_id"] != run_id or not acknowledgement["executions"]:
                raise AssertionError("Stop acknowledgement belongs to another task")
            if any(item["state"] != "ended" or not item["device_confirmed"] for item in acknowledgement["executions"]):
                raise AssertionError("Native execution lacks a confirmed terminal boundary")
            await asyncio.sleep(2)
            response = await client.get(f"/api/runs/{run_id}?events=none")
            response.raise_for_status()
            after = response.json()
            if [(item["execution_id"], item["control_steps"], item["raw_sim_steps"])
                    for item in after["executions"]] != [(item["execution_id"], item["control_steps"], item["raw_sim_steps"])
                    for item in acknowledgement["executions"]]:
                raise AssertionError("Physical actions continued after stop confirmation")
        await session.send("device-status", "status")
        device = await session.next("voice.status")
        if device["payload"]["playback_active"]:
            raise AssertionError("Audio device remained active after interruption")
        await session.close()
        if any(event["kind"] == "voice.playback.started" and event["request_id"] == "voice-command"
                for event in session.events):
            raise AssertionError("Interrupted task feedback was played")
        result = {"passed": True, "scope": "Actual Qwen ASR, model task, simulation actions and native execution interruption",
            "input_source": "Provided recorded WAV; live microphone capture separate",
            "physical_braking": "Metric tool stopping acceptance is separate from execution-clock interruption",
            "transcription": transcription, "before": before, "stop": stopped, "after": after,
            "device": device, "events": session.events}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
            allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"passed": True, "run_id": run_id,
            "executions": acknowledgement["executions"]}, ensure_ascii=False), flush=True)
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
            await process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--asr-url", default="http://127.0.0.1:18761")
    parser.add_argument("--tts-url", default="http://127.0.0.1:18762")
    parser.add_argument("--harness-url", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
