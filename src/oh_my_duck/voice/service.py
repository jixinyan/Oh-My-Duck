import argparse
import asyncio
from io import BytesIO
import logging
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
import soundfile as sf
import torch
import uvicorn

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.audio import _sha256, _source_path, _validate_wav
from oh_my_duck.voice.profiles import SQLiteVoiceProfileStore


class SpeechRequest(BaseModel):
    persona_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _audio_info(content: bytes):
    if len(content) == 0 or len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "WAV 大小必须在 1 字节至 20 MiB 之间")
    info = sf.info(BytesIO(content))
    if info.format != "WAV" or info.channels not in {1, 2} or info.frames <= 0 or info.duration > 60:
        raise HTTPException(422, "需要不超过 60 秒的有效 WAV")
    return info


def create_asr_app(model, inbox: Path) -> FastAPI:
    inbox = inbox.resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Oh My Duck ASR")

    @app.get("/health")
    def health() -> dict:
        return {"service": "asr", "model_revision": model.model_revision}

    @app.post("/v1/transcriptions")
    async def transcribe(request: Request, language_hint: str | None = None) -> dict:
        content = await request.body()
        _audio_info(content)
        path = inbox / f"{uuid4().hex}.wav"
        with path.open("xb") as output:
            output.write(content)
        try:
            _validate_wav(path)
            audio = PayloadRef(path.as_uri(), "audio/wav", _sha256(path))
            text = await model.transcribe(audio, language_hint=language_hint)
            return {"text": text, "model_revision": model.model_revision, "audio_sha256": audio.sha256}
        finally:
            path.unlink(missing_ok=True)

    return app


def create_tts_app(model, store: SQLiteVoiceProfileStore) -> FastAPI:
    app = FastAPI(title="Oh My Duck TTS")

    @app.get("/health")
    def health() -> dict:
        return {"service": "tts", "model_revision": model.model_revision}

    @app.post("/v1/speech")
    async def speech(request: SpeechRequest) -> Response:
        profile = await asyncio.to_thread(store.active, request.persona_id)
        if profile is None:
            raise HTTPException(404, "该 persona 没有已确认的活动音色")
        if profile.synthesis_model_revision != model.model_revision:
            raise HTTPException(409, "活动音色与服务模型 revision 不一致")
        logging.getLogger("uvicorn.error").info(
            "voice.speech.started persona_id=%s profile_revision=%s",
            profile.persona_id, profile.revision,
        )
        audio = await model.synthesize(request.text, profile)
        path = _source_path(audio.uri)
        _validate_wav(path)
        content = path.read_bytes()
        return Response(
            content=content, media_type="audio/wav",
            headers={
                "X-Audio-SHA256": audio.sha256,
                "X-Voice-Persona-ID": quote(profile.persona_id, safe=""),
                "X-Voice-ID": quote(profile.voice_id, safe=""),
                "X-Voice-Profile-Revision": str(profile.revision),
                "X-Voice-Model-Revision": model.model_revision,
            },
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="本机回环地址上的独立 Qwen 语音服务")
    parser.add_argument("kind", choices=("asr", "tts"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.kind == "asr":
        from oh_my_duck.voice.qwen import QwenSpeechRecognition

        model = QwenSpeechRecognition(device=args.device, revision=args.model_revision)
        app = create_asr_app(model, args.data_dir / "inbox")
    else:
        from oh_my_duck.voice.qwen import QwenSpeechSynthesis

        store = SQLiteVoiceProfileStore(args.data_dir)
        model = QwenSpeechSynthesis(
            args.data_dir / "output", device=args.device, revision=args.model_revision
        )
        app = create_tts_app(model, store)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
