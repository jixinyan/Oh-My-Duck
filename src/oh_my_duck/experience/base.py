from typing import Protocol
from oh_my_duck.core.contracts.events import EpisodeEvent


class EpisodeRecorder(Protocol):
    def append(self, event: EpisodeEvent) -> None: ...
