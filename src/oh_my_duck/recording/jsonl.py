"""Single-process JSONL recorder; consumers must not replay records as robot commands."""
from dataclasses import asdict
import json
from pathlib import Path
from threading import Lock

from ..contracts.events import EpisodeEvent


class JsonlEpisodeRecorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: EpisodeEvent) -> None:
        line = json.dumps(asdict(event), ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
