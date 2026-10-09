import hashlib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import soundfile as sf

from oh_my_duck.core.contracts.sensors import PayloadRef


_WAV_SUBTYPES = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT"})


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _source_path(uri: str) -> Path:
    _required(uri, "reference_audio.uri")
    if uri.startswith("file:"):
        parsed = urlsplit(uri)
        if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("仅支持本地 file URI")
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri)
    if not path.is_absolute():
        raise ValueError("参考音频必须使用本地绝对路径")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_wav(path: Path) -> None:
    info = sf.info(path)
    if info.format != "WAV" or info.subtype not in _WAV_SUBTYPES:
        raise ValueError("参考音频必须是受支持的 WAV 编码")
    if info.channels not in {1, 2} or info.samplerate <= 0 or info.frames <= 0:
        raise ValueError("参考音频必须包含单声道或双声道音频帧")
    with sf.SoundFile(path) as audio:
        frames = 0
        while True:
            block = audio.read(65536, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            if not np.isfinite(block).all():
                raise ValueError("参考音频包含非有限采样值")
            frames += len(block)
        if frames != info.frames:
            raise ValueError("参考音频帧不完整")


def _audio_ref(path: Path) -> PayloadRef:
    _validate_wav(path)
    return PayloadRef(uri=path.resolve().as_uri(), media_type="audio/wav", sha256=_sha256(path))


def _write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> PayloadRef:
    if sample_rate <= 0 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("模型生成了无效音频")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as output:
        sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    path.chmod(0o600)
    return _audio_ref(path)
