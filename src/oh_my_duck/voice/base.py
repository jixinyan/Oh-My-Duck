from dataclasses import dataclass
from typing import Protocol

from ..contracts.sensors import PayloadRef


@dataclass(frozen=True)
class VoiceProfile:
    persona_id: str
    voice_id: str
    revision: int
    reference_audio: PayloadRef
    reference_text: str
    design_model_revision: str
    synthesis_model_revision: str
    confirmed_at: str


@dataclass(frozen=True)
class VoiceCandidate:
    candidate_id: str
    description: str
    reference_audio: PayloadRef
    reference_text: str
    model_revision: str


class VoiceDesign(Protocol):
    async def design(self, description: str, reference_text: str) -> VoiceCandidate: ...


class VoiceProfileStore(Protocol):
    def active(self, persona_id: str) -> VoiceProfile | None: ...
    def save_confirmed(self, profile: VoiceProfile) -> None: ...


class SpeechRecognition(Protocol):
    async def transcribe(self, audio: PayloadRef, *, language_hint: str | None = None) -> str: ...


class SpeechSynthesis(Protocol):
    async def synthesize(self, text: str, profile: VoiceProfile) -> PayloadRef: ...


class AudioDevice(Protocol):
    async def begin_recording(self) -> str: ...
    async def finish_recording(self, recording_id: str) -> PayloadRef: ...
    async def play(self, audio: PayloadRef) -> str: ...
    async def stop_playback(self, playback_id: str) -> None: ...
