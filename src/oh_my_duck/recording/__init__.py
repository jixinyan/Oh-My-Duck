"""Append-only episode evidence. Summarization and long-term memory stay outside this package."""
from .jsonl import JsonlEpisodeRecorder
__all__ = ["JsonlEpisodeRecorder"]
