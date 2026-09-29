import asyncio
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

import httpx

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.profiles import _sha256, _source_path, _validate_wav


@dataclass(frozen=True)
class Transcription:
    text: str
    model_revision: str
    audio_sha256: str


@dataclass(frozen=True)
class SpeechAudio:
    audio: PayloadRef
    persona_id: str
    voice_id: str
    profile_revision: int
    model_revision: str


class RemoteVoiceServices:
    def __init__(self, asr_url: str, tts_url: str, output_dir: str | Path):
        for url in (asr_url, tts_url):
            parsed = httpx.URL(url)
            if parsed.scheme != "http" or parsed.host not in {"127.0.0.1", "localhost"}:
                raise ValueError("语音服务只能通过本机回环地址连接")
        self.asr_url = asr_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def transcribe(self, audio: PayloadRef, *, language_hint: str | None = None) -> Transcription:
        path = _source_path(audio.uri)
        await asyncio.to_thread(_validate_wav, path)
        content = await asyncio.to_thread(path.read_bytes)
        digest = await asyncio.to_thread(_sha256, path)
        if audio.sha256 is not None and digest != audio.sha256:
            raise ValueError("录音 SHA256 不匹配")
        response = await self._client.post(
            f"{self.asr_url}/v1/transcriptions", content=content,
            params={} if language_hint is None else {"language_hint": language_hint},
            headers={"Content-Type": "audio/wav"},
        )
        response.raise_for_status()
        data = response.json()
        if data["audio_sha256"] != digest or not data["text"].strip() or not data["model_revision"]:
            raise ValueError("ASR 响应与请求录音不一致")
        return Transcription(data["text"], data["model_revision"], digest)

    async def synthesize(self, persona_id: str, text: str) -> SpeechAudio:
        if not persona_id.strip() or not text.strip():
            raise ValueError("persona_id 和 text 必须有内容")
        response = await self._client.post(
            f"{self.tts_url}/v1/speech", json={"persona_id": persona_id, "text": text}
        )
        response.raise_for_status()
        headers = response.headers
        if unquote(headers["x-voice-persona-id"]) != persona_id:
            raise ValueError("TTS 返回了其他 persona 的音色")
        revision = int(headers["x-voice-profile-revision"])
        voice_id = unquote(headers["x-voice-id"])
        if revision < 1 or not voice_id or not headers["x-voice-model-revision"]:
            raise ValueError("TTS 音色 metadata 无效")
        path = self.output_dir / f"{uuid4().hex}.wav"
        with path.open("xb") as output:
            output.write(response.content)
        _validate_wav(path)
        digest = _sha256(path)
        if digest != headers["x-audio-sha256"]:
            raise ValueError("TTS 音频 SHA256 不匹配")
        return SpeechAudio(
            PayloadRef(path.as_uri(), "audio/wav", digest), persona_id,
            voice_id, revision, headers["x-voice-model-revision"],
        )

    async def close(self) -> None:
        await self._client.aclose()
