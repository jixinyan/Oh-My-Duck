import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.base import VoiceCandidate, VoiceProfile
from oh_my_duck.voice.profiles import SQLiteVoiceProfileStore


REFERENCE_SHA256 = "480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c"
REFERENCE_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_audio() -> Path:
    path = Path(os.environ.get(
        "OH_MY_DUCK_VOICE_REFERENCE_WAV",
        PROJECT_ROOT / ".cache/voice-profiles/reference.wav",
    ))
    if not path.is_file():
        pytest.skip("需要下载官方 Qwen3-TTS 参考音频到 .cache/voice-profiles/reference.wav")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    return path


@pytest.fixture
def store_root():
    root = PROJECT_ROOT / ".cache/voice-profiles/tests" / uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)


def profile(path: Path, *, persona_id: str = "duck-001", revision: int = 1, voice_id: str = "voice-001") -> VoiceProfile:
    return VoiceProfile(
        persona_id=persona_id,
        voice_id=voice_id,
        revision=revision,
        reference_audio=PayloadRef(uri=path.as_uri(), media_type="audio/wav", sha256=REFERENCE_SHA256),
        reference_text=REFERENCE_TEXT,
        design_model_revision="Qwen3-TTS-VoiceDesign@revision-1",
        synthesis_model_revision="Qwen3-TTS-Base@revision-1",
        confirmed_at="2026-09-23T12:00:00-05:00",
    )


def managed_path(saved: VoiceProfile) -> Path:
    return Path(unquote(urlsplit(saved.reference_audio.uri).path))


def test_persists_managed_audio_across_store_instances(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    store.save_confirmed(profile(reference_audio))
    saved = SQLiteVoiceProfileStore(store_root).active("duck-001")
    assert saved is not None
    assert saved.reference_audio.uri != reference_audio.as_uri()
    assert managed_path(saved).read_bytes() == reference_audio.read_bytes()
    assert saved.reference_audio.sha256 == REFERENCE_SHA256
    assert saved.reference_text == REFERENCE_TEXT


def test_candidate_requires_confirmation_and_survives_restart(reference_audio: Path, store_root: Path):
    candidate = VoiceCandidate(
        candidate_id="candidate-001",
        description="温暖的中文声音",
        reference_audio=PayloadRef(reference_audio.as_uri(), "audio/wav", REFERENCE_SHA256),
        reference_text=REFERENCE_TEXT,
        model_revision="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@revision-1",
    )
    store = SQLiteVoiceProfileStore(store_root)
    store.save_candidate(candidate)
    assert store.active("duck-001") is None
    recovered = SQLiteVoiceProfileStore(store_root).candidate(candidate.candidate_id)
    assert recovered.reference_audio.sha256 == REFERENCE_SHA256
    assert managed_path(recovered).is_file()
    confirmed = store.confirm_candidate(
        candidate.candidate_id, "duck-001", "Qwen/Qwen3-TTS-12Hz-0.6B-Base@revision-1"
    )
    assert confirmed.revision == 1
    assert confirmed.voice_id == candidate.candidate_id
    assert SQLiteVoiceProfileStore(store_root).active("duck-001") == confirmed


def test_personas_and_history_are_independent(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    store.save_confirmed(profile(reference_audio))
    store.save_confirmed(profile(reference_audio, persona_id="duck-002"))
    store.save_confirmed(profile(reference_audio, revision=2, voice_id="voice-002"))
    assert store.active("duck-001").revision == 2
    assert store.active("duck-002").revision == 1
    assert [item.revision for item in store.history("duck-001")] == [1, 2]
    assert store.get("duck-001", 1).voice_id == "voice-001"
    assert store.get("duck-001", 3) is None


def test_idempotent_content_does_not_reactivate_old_revision(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    original = profile(reference_audio)
    store.save_confirmed(original)
    store.save_confirmed(profile(reference_audio, revision=2, voice_id="voice-002"))
    copied_source = store_root / "same-reference.wav"
    shutil.copyfile(reference_audio, copied_source)
    store.save_confirmed(replace(original, reference_audio=replace(original.reference_audio, uri=copied_source.as_uri())))
    assert store.active("duck-001").revision == 2
    assert len(store.history("duck-001")) == 2


def test_rejected_updates_preserve_active(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    store.save_confirmed(profile(reference_audio))
    next_profile = profile(reference_audio, revision=2, voice_id="voice-002")
    with pytest.raises(ValueError, match="不同内容"):
        store.save_confirmed(replace(profile(reference_audio), voice_id="other"))
    with pytest.raises(ValueError, match="revision 必须是 2"):
        store.save_confirmed(profile(reference_audio, revision=3))
    with pytest.raises(ValueError, match="SHA256 不匹配"):
        store.save_confirmed(replace(next_profile, reference_audio=replace(next_profile.reference_audio, sha256="0" * 64)))
    with pytest.raises(FileNotFoundError):
        store.save_confirmed(replace(next_profile, reference_audio=replace(next_profile.reference_audio, uri=(store_root / "missing.wav").as_uri())))
    with pytest.raises(ValueError, match="本地绝对路径"):
        store.save_confirmed(replace(next_profile, reference_audio=replace(next_profile.reference_audio, uri="https://example.com/voice.wav")))
    assert store.active("duck-001").revision == 1
    assert [item.revision for item in store.history("duck-001")] == [1]


def test_rejects_incomplete_metadata(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    original = profile(reference_audio)
    for invalid in (
        replace(original, persona_id=" "),
        replace(original, design_model_revision=""),
        replace(original, reference_text=""),
        replace(original, confirmed_at="2026-09-23T12:00:00"),
    ):
        with pytest.raises(ValueError):
            store.save_confirmed(invalid)
    assert store.active("duck-001") is None


def test_detects_missing_and_changed_managed_audio(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    store.save_confirmed(profile(reference_audio))
    stored_path = managed_path(store.active("duck-001"))
    stored_path.unlink()
    with pytest.raises(FileNotFoundError):
        store.active("duck-001")
    store.save_confirmed(profile(reference_audio, revision=2, voice_id="voice-002"))
    stored_path = managed_path(store.active("duck-001"))
    stored_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 不匹配"):
        store.get("duck-001", 1)


def test_rejects_changed_repeat_source(reference_audio: Path, store_root: Path):
    store = SQLiteVoiceProfileStore(store_root)
    original = profile(reference_audio)
    store.save_confirmed(original)
    changed_source = store_root / "changed.wav"
    changed_source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 不匹配"):
        store.save_confirmed(replace(original, reference_audio=replace(original.reference_audio, uri=changed_source.as_uri())))
    assert store.active("duck-001").revision == 1


def test_rejects_nonfinite_audio(reference_audio: Path, store_root: Path):
    samples, sample_rate = sf.read(reference_audio, dtype="float32")
    samples[0] = np.nan
    changed_source = store_root / "nonfinite.wav"
    sf.write(changed_source, samples, sample_rate, subtype="FLOAT")
    altered = profile(changed_source)
    altered = replace(
        altered,
        reference_audio=replace(
            altered.reference_audio,
            sha256=hashlib.sha256(changed_source.read_bytes()).hexdigest(),
        ),
    )
    store = SQLiteVoiceProfileStore(store_root)
    with pytest.raises(ValueError, match="非有限采样值"):
        store.save_confirmed(altered)
    assert store.active("duck-001") is None


def test_concurrent_conflicting_revision_has_one_winner(reference_audio: Path, store_root: Path):
    SQLiteVoiceProfileStore(store_root)
    barrier = Barrier(2)

    def save(voice_id: str):
        store = SQLiteVoiceProfileStore(store_root)
        barrier.wait()
        store.save_confirmed(profile(reference_audio, voice_id=voice_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda voice_id: _capture_save(save, voice_id), ("voice-a", "voice-b")))
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert SQLiteVoiceProfileStore(store_root).active("duck-001").voice_id in {"voice-a", "voice-b"}


def _capture_save(save, voice_id: str):
    try:
        save(voice_id)
    except ValueError as error:
        return error
    return None
