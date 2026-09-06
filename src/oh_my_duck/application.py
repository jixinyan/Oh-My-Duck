"""Explicit dependency injection for future UI/services; no implicit agent loop."""
from dataclasses import dataclass

from .backends import RobotBackend
from .harness import HarnessBridge
from .perception import ActivePerception
from .recording.base import EpisodeRecorder
from .skills import SkillRunner
from .tools import ToolCatalog
from .voice import AudioDevice, SpeechRecognition, SpeechSynthesis, VoiceDesign, VoiceProfileStore


@dataclass
class ApplicationServices:
    tools: ToolCatalog
    recorder: EpisodeRecorder
    backend: RobotBackend | None = None
    skills: SkillRunner | None = None
    perception: ActivePerception | None = None
    harness: HarnessBridge | None = None
    asr: SpeechRecognition | None = None
    tts: SpeechSynthesis | None = None
    voice_design: VoiceDesign | None = None
    voice_profiles: VoiceProfileStore | None = None
    audio: AudioDevice | None = None
