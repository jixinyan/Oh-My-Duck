"""Explicit dependency injection for future UI/services; no implicit agent loop."""
from dataclasses import dataclass

from oh_my_duck.robotics.backends import RobotBackend
from oh_my_duck.agentic.harness import HarnessBridge
from oh_my_duck.perception import ActivePerception
from oh_my_duck.experience.base import EpisodeRecorder
from oh_my_duck.agentic.skills import SkillRunner
from oh_my_duck.agentic.tools import ToolCatalog
from oh_my_duck.voice import AudioDevice, SpeechRecognition, SpeechSynthesis, VoiceDesign, VoiceProfileStore


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
