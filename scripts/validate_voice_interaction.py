import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import signal
import sys
from urllib.parse import unquote, urlsplit

import numpy as np
import sounddevice as sd
import soundfile as sf

from oh_my_duck.voice.device import PortAudioDevice
from oh_my_duck.voice.remote import RemoteVoiceServices


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def external_play(audio_path: Path, output_device: str) -> None:
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    sd.play(samples, sample_rate, device=output_device)
    sd.wait()


async def remote_text(host: str, command: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", host, command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode())
    return stdout.decode().strip()


async def remote_output_names(host: str, directory: str) -> set[str]:
    listing = await remote_text(
        host,
        f"find {shlex.quote(directory)} -maxdepth 1 -type f -name '*.wav' -printf '%f\\n'",
    )
    return set(listing.splitlines())


async def remote_start_count(host: str, log_path: str) -> int:
    return int(await remote_text(
        host, f"grep -c 'voice.speech.started' {shlex.quote(log_path)}"
    ))


class SessionProcess:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self.events: list[dict] = []

    async def next(self, expected: str) -> dict:
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=600)
            if not line:
                raise RuntimeError((await self.process.stderr.read()).decode())
            event = json.loads(line)
            self.events.append(event)
            if event["kind"] in {"voice.request.error", "voice.playback.error", "voice.recording.error"}:
                raise RuntimeError(json.dumps(event, ensure_ascii=False))
            if event["kind"] == expected:
                return event

    async def send(self, request_id: str, command: str, **parameters) -> None:
        line = json.dumps({"request_id": request_id, "command": command, **parameters}, ensure_ascii=False)
        self.process.stdin.write((line + "\n").encode())
        await self.process.stdin.drain()

    async def close(self) -> None:
        self.process.stdin.close()
        await self.next("voice.session.closed")
        if await self.process.wait() != 0:
            raise RuntimeError((await self.process.stderr.read()).decode())


async def start_session(args, directory: Path) -> SessionProcess:
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(PROJECT_ROOT / "omd.py"), "voice-session",
        "--persona", args.persona, "--robot-id", args.robot_id, "--domain", "real",
        "--input-device", args.input_device, "--output-device", args.output_device,
        "--asr-url", args.asr_url, "--tts-url", args.tts_url,
        "--data-dir", str(directory), "--log", str(directory / "events.jsonl"),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    session = SessionProcess(process)
    await session.next("voice.session.started")
    return session


async def validate(args) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    services = RemoteVoiceServices(args.asr_url, args.tts_url, output_dir / "speech")
    try:
        speech = await services.synthesize(args.persona, args.text)
        direct_asr = await services.transcribe(speech.audio)
    finally:
        await services.close()

    device = PortAudioDevice(
        output_dir / "device", input_device=args.input_device, output_device=args.output_device
    )
    try:
        short_playback = await device.play(speech.audio)
        stopped_outcome = await device.stop_playback(short_playback)
        full_playback = await device.play(speech.audio)
        completed_outcome = await device.wait_playback(full_playback)
        if stopped_outcome != "stopped" or completed_outcome != "completed":
            raise RuntimeError("播放终态与实际操作不一致")
    finally:
        await device.close()

    session_dir = output_dir / "session"
    session = await start_session(args, session_dir)
    try:
        await session.send("record-start", "begin_recording")
        started = await session.next("voice.recording.started")
        player = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), "--external-play",
            str(Path(unquote(urlsplit(speech.audio.uri).path))),
            "--output-device", args.output_device,
            cwd=PROJECT_ROOT,
        )
        if await player.wait() != 0:
            raise RuntimeError("真实扬声器播放进程失败")
        await asyncio.sleep(0.2)
        await session.send(
            "record-finish", "finish_recording", recording_id=started["payload"]["recording_id"]
        )
        finished = await session.next("voice.recording.finished")
        recorded_path = Path(unquote(urlsplit(finished["payload"]["audio"]).path))
        recorded_samples, sample_rate = sf.read(recorded_path, dtype="float32")
        await session.send("acoustic-asr", "transcribe")
        await session.next("voice.request.accepted")
        acoustic_asr = await session.next("voice.transcription.completed")

        before_start = await remote_start_count(args.ssh_host, args.remote_tts_log)
        before_files = await remote_output_names(args.ssh_host, args.remote_tts_output)
        await session.send("cancel-speak", "speak", text=args.cancel_text)
        await session.next("voice.request.accepted")
        for _ in range(40):
            if await remote_start_count(args.ssh_host, args.remote_tts_log) > before_start:
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError("远端 TTS 未确认开始推理")
        await session.send("cancel-stop", "stop")
        await session.next("voice.stopped")
        await session.next("voice.request.cancelled")
        for _ in range(120):
            new_files = await remote_output_names(args.ssh_host, args.remote_tts_output) - before_files
            if new_files:
                break
            await asyncio.sleep(2)
        else:
            raise RuntimeError("远端 TTS 未完成中断后的实际推理")
        await session.send("cancel-status", "status")
        status = await session.next("voice.status")
        if status["payload"]["playback_id"] is not None:
            raise RuntimeError("中断后的音频仍在播放")
        await session.close()
        if any(
            event["kind"] == "voice.playback.started" and event["request_id"] == "cancel-speak"
            for event in session.events
        ):
            raise RuntimeError("中断后的合成结果被播放")
        result = {
            "speech": {
                "audio": speech.audio.uri, "audio_sha256": speech.audio.sha256,
                "persona_id": speech.persona_id, "voice_id": speech.voice_id,
                "profile_revision": speech.profile_revision, "model_revision": speech.model_revision,
            },
            "direct_asr": asdict(direct_asr),
            "playback": {"stopped": stopped_outcome, "completed": completed_outcome},
            "acoustic_recording": {
                "audio": finished["payload"]["audio"],
                "audio_sha256": finished["payload"]["audio_sha256"],
                "sample_rate": sample_rate, "frames": len(recorded_samples),
                "rms": float(np.sqrt(np.mean(recorded_samples ** 2))),
                "peak": float(np.max(np.abs(recorded_samples))),
            },
            "acoustic_asr": acoustic_asr["payload"],
            "cancel": {"request_id": "cancel-speak", "remote_output_files": sorted(new_files),
                       "status": status["payload"]},
            "session_log": str(session_dir / "events.jsonl"),
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if session.process.returncode is None:
            session.process.send_signal(signal.SIGINT)
            await session.process.wait()


async def validate_session_playback(args) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    session_dir = output_dir / "session"
    session = await start_session(args, session_dir)
    try:
        await session.send("complete-speak", "speak", text=args.text)
        await session.next("voice.request.accepted")
        started = await session.next("voice.playback.started")
        completed = await session.next("voice.playback.completed")
        if started["request_id"] != "complete-speak" or completed["request_id"] != "complete-speak":
            raise RuntimeError("自然播放事件对应错误请求")
        if started["payload"]["playback_id"] != completed["payload"]["playback_id"]:
            raise RuntimeError("自然播放事件对应错误句柄")

        await session.send("stop-speak", "speak", text=args.cancel_text)
        await session.next("voice.request.accepted")
        stop_started = await session.next("voice.playback.started")
        if stop_started["request_id"] != "stop-speak":
            raise RuntimeError("中断播放事件对应错误请求")
        await session.send("playback-stop", "stop")
        stopped = await session.next("voice.stopped")
        if stopped["payload"].get("playback_id") != stop_started["payload"]["playback_id"]:
            raise RuntimeError("中断播放对应错误句柄")
        if stopped["payload"].get("playback_outcome") != "stopped":
            raise RuntimeError("播放中断没有得到设备停止确认")
        await session.send("playback-status", "status")
        status = await session.next("voice.status")
        if status["payload"]["playback_active"]:
            raise RuntimeError("播放中断后设备仍在运行")
        await session.close()
        if any(
            event["kind"] == "voice.playback.completed" and event["request_id"] == "stop-speak"
            for event in session.events
        ):
            raise RuntimeError("中断播放出现自然完成事件")
        result = {
            "natural_playback": {
                "request_id": "complete-speak", "started": started["payload"],
                "completed": completed["payload"],
            },
            "stopped_playback": {
                "request_id": "stop-speak", "started": stop_started["payload"],
                "stopped": stopped["payload"], "status": status["payload"],
            },
            "session_log": str(session_dir / "events.jsonl"),
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if session.process.returncode is None:
            session.process.send_signal(signal.SIGINT)
            await session.process.wait()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--external-play":
        parser = argparse.ArgumentParser()
        parser.add_argument("--external-play", type=Path, required=True)
        parser.add_argument("--output-device", required=True)
        external_args = parser.parse_args()
        external_play(external_args.external_play, external_args.output_device)
        return 0
    parser = argparse.ArgumentParser(description="真实设备与远端常驻模型语音验收")
    parser.add_argument("--persona", required=True)
    parser.add_argument("--mode", choices=("full", "session-playback"), default="full")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--asr-url", required=True)
    parser.add_argument("--tts-url", required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--remote-tts-log")
    parser.add_argument("--remote-tts-output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default="你好，我是小鸭。我们现在检查语音连接。")
    parser.add_argument("--cancel-text", default="你好，我们正在检查语音合成任务被中断以后的播放状态。")
    args = parser.parse_args()
    if args.mode == "full" and not all((args.ssh_host, args.remote_tts_log, args.remote_tts_output)):
        parser.error("full 模式需要 --ssh-host、--remote-tts-log 和 --remote-tts-output")
    result = asyncio.run(validate(args) if args.mode == "full" else validate_session_playback(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
