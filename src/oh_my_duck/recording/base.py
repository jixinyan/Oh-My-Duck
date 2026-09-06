from typing import Protocol
from ..contracts.events import EpisodeEvent


class EpisodeRecorder(Protocol):
    def append(self, event: EpisodeEvent) -> None: ...
