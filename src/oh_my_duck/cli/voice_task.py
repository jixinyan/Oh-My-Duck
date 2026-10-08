import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.integrations.native_client import NativeTaskClient
from oh_my_duck.voice.remote import RemoteVoiceServices


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    services = RemoteVoiceServices(args.asr_url, args.tts_url, args.output / "speech")
    tasks = NativeTaskClient(args.harness_url, args.profile, args.scenario, args.output / "native")
    result = {"scope": "Recorded WAV → Qwen ASR → native Harness → confirmed-voice Qwen TTS",
        "live_microphone": False, "speaker_playback": False}
    try:
        audio = PayloadRef(args.audio.resolve(strict=True).as_uri(), "audio/wav")
        transcription = await services.transcribe(audio, language_hint=args.language_hint)
        result["transcription"] = asdict(transcription)
        await tasks.open()
        submitted = await tasks.submit(transcription.text)
        result["submission"] = submitted
        print(json.dumps({"transcription": asdict(transcription), **submitted}, ensure_ascii=False), flush=True)
        native = await tasks.wait(timeout_s=args.timeout)
        result["native_run"] = native
        text = {"succeeded": "任务已经完成。", "failed": "任务执行失败，请查看任务记录。",
            "cancelled": "任务已经中断。", "interrupted": "任务已经中断。",
            "unknown": "任务状态无法确认，请查看任务记录。"}[native["state"]]
        response = await services.synthesize(args.persona, text)
        result["speech"] = asdict(response)
        print(json.dumps({"runId": native["id"], "state": native["state"],
            "speech": asdict(response)}, ensure_ascii=False), flush=True)
        return 0 if native["state"] == "succeeded" else 2
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False,
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            await tasks.close()
        finally:
            await services.close()


def main():
    parser = argparse.ArgumentParser(description="录音指令、原生任务和固定音色反馈")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--harness-url", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--asr-url", default="http://127.0.0.1:18761")
    parser.add_argument("--tts-url", default="http://127.0.0.1:18762")
    parser.add_argument("--language-hint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=2400)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout 必须是有限的正数")
    return asyncio.run(run(args))
