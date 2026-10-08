import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlsplit

import soundfile as sf

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.profiles import SQLiteVoiceProfileStore


def _local_path(uri: str) -> str:
    return unquote(urlsplit(uri).path)


def _peak_cuda_memory_bytes(device: str) -> int | None:
    import torch

    return None if device == "cpu" else torch.cuda.max_memory_reserved(device=device)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("cpu-threads 必须是正整数")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen 语音识别、音色确认和语音合成")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".local/share/oh-my-duck/voice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="将本地 WAV 录音转为文本")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--language")
    transcribe.add_argument("--model-revision")
    design = commands.add_parser("design", help="根据描述生成试听音色")
    design.add_argument("--description", required=True)
    design.add_argument("--reference-text", required=True)
    design.add_argument("--language", default="Auto")
    confirm = commands.add_parser("confirm", help="确认候选音色并设为活动音色")
    confirm.add_argument("candidate_id")
    confirm.add_argument("--persona", required=True)
    active = commands.add_parser("active", help="查看活动音色")
    active.add_argument("--persona", required=True)
    synthesize = commands.add_parser("synthesize", help="使用活动音色合成语音")
    synthesize.add_argument("--persona", required=True)
    synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--language", default="Auto")
    args = parser.parse_args()
    store = SQLiteVoiceProfileStore(args.data_dir)

    if args.command == "transcribe":
        import torch

        from oh_my_duck.voice.qwen import QwenSpeechRecognition

        torch.set_num_threads(args.cpu_threads)
        audio = args.audio.expanduser().resolve(strict=True)
        started = perf_counter()
        recognizer = QwenSpeechRecognition(device=args.device, revision=args.model_revision)
        loaded = perf_counter()
        result = asyncio.run(recognizer.transcribe(
            PayloadRef(uri=audio.as_uri(), media_type="audio/wav"), language_hint=args.language
        ))
        print(json.dumps({
            "text": result, "audio_duration_seconds": sf.info(audio).duration,
            "model_revision": recognizer.model_revision,
            "load_seconds": loaded - started, "inference_seconds": perf_counter() - loaded,
            "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(args.device),
        }, ensure_ascii=False))
    elif args.command == "design":
        import torch

        from oh_my_duck.voice.qwen import QwenVoiceDesign

        torch.set_num_threads(args.cpu_threads)
        started = perf_counter()
        designer = QwenVoiceDesign(args.data_dir / "candidates", device=args.device, language=args.language)
        loaded = perf_counter()
        candidate = asyncio.run(designer.design(args.description, args.reference_text))
        store.save_candidate(candidate)
        print(json.dumps({
            "candidate_id": candidate.candidate_id,
            "audio": _local_path(candidate.reference_audio.uri),
            "reference_text": candidate.reference_text,
            "model_revision": candidate.model_revision,
            "audio_duration_seconds": sf.info(_local_path(candidate.reference_audio.uri)).duration,
            "load_seconds": loaded - started, "inference_seconds": perf_counter() - loaded,
            "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(args.device),
        }, ensure_ascii=False))
    elif args.command == "confirm":
        from huggingface_hub import HfApi

        from oh_my_duck.voice.qwen import BASE_MODEL

        revision = HfApi().model_info(BASE_MODEL).sha
        profile = store.confirm_candidate(args.candidate_id, args.persona, f"{BASE_MODEL}@{revision}")
        print(json.dumps({
            "persona_id": profile.persona_id, "voice_id": profile.voice_id,
            "revision": profile.revision, "reference_audio": _local_path(profile.reference_audio.uri),
        }, ensure_ascii=False))
    elif args.command == "active":
        profile = store.active(args.persona)
        if profile is None:
            raise ValueError("该 persona 没有活动音色")
        print(json.dumps({
            "persona_id": profile.persona_id, "voice_id": profile.voice_id,
            "revision": profile.revision, "reference_audio": _local_path(profile.reference_audio.uri),
            "reference_text": profile.reference_text,
            "design_model_revision": profile.design_model_revision,
            "synthesis_model_revision": profile.synthesis_model_revision,
        }, ensure_ascii=False))
    else:
        import torch

        from oh_my_duck.voice.qwen import BASE_MODEL, QwenSpeechSynthesis

        torch.set_num_threads(args.cpu_threads)
        profile = store.active(args.persona)
        if profile is None:
            raise ValueError("该 persona 没有活动音色")
        model, revision = profile.synthesis_model_revision.split("@", maxsplit=1)
        if model != BASE_MODEL:
            raise ValueError("活动音色的合成模型不受支持")
        started = perf_counter()
        synthesizer = QwenSpeechSynthesis(
            args.data_dir / "output", device=args.device, revision=revision, language=args.language
        )
        loaded = perf_counter()
        audio = asyncio.run(synthesizer.synthesize(args.text, profile))
        print(json.dumps({
            "audio": _local_path(audio.uri), "sha256": audio.sha256,
            "audio_duration_seconds": sf.info(_local_path(audio.uri)).duration,
            "model_revision": synthesizer.model_revision,
            "load_seconds": loaded - started, "inference_seconds": perf_counter() - loaded,
            "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(args.device),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
