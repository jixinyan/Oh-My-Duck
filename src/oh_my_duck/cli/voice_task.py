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
        "live_microphone": False, "speaker_playback": False, "tasks": []}
    try:
        for path in args.audio:
            audio = PayloadRef(path.resolve(strict=True).as_uri(), "audio/wav")
            transcription = await services.transcribe(audio, language_hint=args.language_hint)
            context = () if not args.context_previous_task or tasks.run_id is None else (tasks.run_id,)
            record = {"transcription": asdict(transcription), "context_run_ids": list(context)}
            result["tasks"].append(record)
            if tasks.session_id is None:
                await tasks.open()
            submitted = await tasks.submit(transcription.text, context_run_ids=context)
            record["submission"] = submitted
            print(json.dumps({**record, **submitted}, ensure_ascii=False), flush=True)
            native = await tasks.wait(timeout_s=args.timeout)
            record["native_run"] = native
            text = {"succeeded": "任务已经完成。", "failed": "任务执行失败，请查看任务记录。",
                "cancelled": "任务已经中断。", "interrupted": "任务已经中断。",
                "unknown": "任务状态无法确认，请查看任务记录。"}[native["state"]]
            response = await services.synthesize(args.persona, text)
            record["speech"] = asdict(response)
            result.update(record)
            print(json.dumps({"runId": native["id"], "state": native["state"],
                "speech": asdict(response)}, ensure_ascii=False), flush=True)
            if native["state"] != "succeeded":
                return 2
        return 0
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False,
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            await tasks.close()
        finally:
            await services.close()


def main():
    parser = argparse.ArgumentParser(description="录音指令、原生任务和固定音色反馈")
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--context-previous-task", action="store_true",
                        help="每项后续录音明确引用同一会话中上一项已结束任务")
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
    if args.context_previous_task and len(args.audio) < 2:
        parser.error("context-previous-task 需要至少两项录音")
    return asyncio.run(run(args))
