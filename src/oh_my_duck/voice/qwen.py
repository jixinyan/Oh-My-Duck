import asyncio
from pathlib import Path
from threading import Lock
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.audio import _sha256, _source_path, _validate_wav, _write_audio
from oh_my_duck.voice.base import VoiceCandidate, VoiceProfile


ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def model_snapshot(model_id: str, revision: str | None = None) -> tuple[str, str]:
    path = Path(snapshot_download(repo_id=model_id, revision=revision))
    return str(path), f"{model_id}@{path.name}"


def _model_options(device: str) -> dict:
    if not isinstance(device, str) or not device:
        raise ValueError("device 必须是 cpu 或 cuda 设备")
    if device == "cpu":
        return {"device_map": "cpu", "dtype": torch.float32}
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        return {"device_map": device, "dtype": torch.bfloat16}
    raise ValueError("device 必须是 cpu 或 cuda 设备")


class QwenSpeechRecognition:
    def __init__(self, *, device: str = "cuda:0", revision: str | None = None):
        options = _model_options(device)
        from qwen_asr import Qwen3ASRModel

        snapshot, self.model_revision = model_snapshot(ASR_MODEL, revision)
        self.model = Qwen3ASRModel.from_pretrained(
            snapshot, max_inference_batch_size=1, max_new_tokens=512, **options
        )
        self._model_lock = Lock()

    async def transcribe(self, audio: PayloadRef, *, language_hint: str | None = None) -> str:
        return await asyncio.to_thread(self._transcribe, audio, language_hint)

    def _transcribe(self, audio: PayloadRef, language_hint: str | None) -> str:
        path = _source_path(audio.uri)
        _validate_wav(path)
        if audio.sha256 is not None and _sha256(path) != audio.sha256:
            raise ValueError("录音 SHA256 不匹配")
        with self._model_lock:
            results = self.model.transcribe(audio=str(path), language=language_hint)
        if len(results) != 1 or not results[0].text.strip():
            raise ValueError("ASR 未返回转写文本")
        return results[0].text


class QwenVoiceDesign:
    def __init__(
        self, output_dir: str | Path, *, device: str = "cuda:0", revision: str | None = None,
        language: str = "Auto",
    ):
        options = _model_options(device)
        from qwen_tts import Qwen3TTSModel

        snapshot, self.model_revision = model_snapshot(DESIGN_MODEL, revision)
        self.model = Qwen3TTSModel.from_pretrained(snapshot, **options)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.language = language
        self._model_lock = Lock()

    async def design(self, description: str, reference_text: str) -> VoiceCandidate:
        return await asyncio.to_thread(self._design, description, reference_text)

    def _design(self, description: str, reference_text: str) -> VoiceCandidate:
        if not description.strip() or not reference_text.strip():
            raise ValueError("description 和 reference_text 必须有内容")
        with self._model_lock:
            waveforms, sample_rate = self.model.generate_voice_design(
                text=reference_text, language=self.language, instruct=description
            )
        if len(waveforms) != 1:
            raise ValueError("VoiceDesign 必须返回一段音频")
        candidate_id = uuid4().hex
        audio = _write_audio(self.output_dir / f"{candidate_id}.wav", waveforms[0], sample_rate)
        return VoiceCandidate(candidate_id, description, audio, reference_text, self.model_revision)


class QwenSpeechSynthesis:
    def __init__(
        self, output_dir: str | Path, *, device: str = "cuda:0", revision: str | None = None,
        language: str = "Auto",
    ):
        options = _model_options(device)
        from qwen_tts import Qwen3TTSModel

        snapshot, self.model_revision = model_snapshot(BASE_MODEL, revision)
        self.model = Qwen3TTSModel.from_pretrained(snapshot, **options)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.language = language
        self._model_lock = Lock()

    async def synthesize(self, text: str, profile: VoiceProfile) -> PayloadRef:
        return await asyncio.to_thread(self._synthesize, text, profile)

    def _synthesize(self, text: str, profile: VoiceProfile) -> PayloadRef:
        if not text.strip():
            raise ValueError("合成文本必须有内容")
        if profile.synthesis_model_revision != self.model_revision:
            raise ValueError("活动音色的合成模型 revision 与当前模型不一致")
        path = _source_path(profile.reference_audio.uri)
        if _sha256(path) != profile.reference_audio.sha256:
            raise ValueError("参考音频 SHA256 不匹配")
        _validate_wav(path)
        with self._model_lock:
            waveforms, sample_rate = self.model.generate_voice_clone(
                text=text, language=self.language, ref_audio=str(path), ref_text=profile.reference_text,
            )
        if len(waveforms) != 1:
            raise ValueError("Base 模型必须返回一段音频")
        return _write_audio(self.output_dir / f"{uuid4().hex}.wav", waveforms[0], sample_rate)
