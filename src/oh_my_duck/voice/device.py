import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

import numpy as np
import sounddevice as sd
import soundfile as sf

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.audio import _sha256, _source_path, _validate_wav


@dataclass
class _Recording:
    identifier: str
    stream: sd.InputStream
    stop_requested: Event = field(default_factory=Event)
    ended: Event = field(default_factory=Event)
    chunks: list[np.ndarray] = field(default_factory=list)
    error: BaseException | None = None
    thread: Thread | None = None


@dataclass
class _Playback:
    identifier: str
    stream: sd.OutputStream
    samples: np.ndarray
    position: int = 0
    stream_lock: Lock = field(default_factory=Lock)
    stop_requested: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)
    closed: bool = False
    outcome: str | None = None
    error: BaseException | None = None


class PortAudioDevice:
    def __init__(
        self, output_dir: str | Path, *, input_device: int | str, output_device: int | str,
        sample_rate: int = 48000, block_frames: int = 1024,
    ):
        if sample_rate <= 0 or block_frames <= 0:
            raise ValueError("sample_rate 和 block_frames 必须是正整数")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.block_frames = block_frames
        sd.check_input_settings(device=input_device, channels=1, samplerate=sample_rate, dtype="float32")
        self._lock = Lock()
        self._recording: _Recording | None = None
        self._recordings: dict[str, _Recording] = {}
        self._playback: _Playback | None = None
        self._playbacks: dict[str, _Playback] = {}
        self._last_error: str | None = None
        self._last_recording_outcome: str | None = None
        self._closed = False

    @staticmethod
    def devices() -> list[dict]:
        return [dict(index=index, **device) for index, device in enumerate(sd.query_devices())]

    def _record_loop(self, recording: _Recording) -> None:
        try:
            while not recording.stop_requested.is_set():
                samples, overflowed = recording.stream.read(self.block_frames)
                if overflowed:
                    raise RuntimeError("录音发生输入溢出")
                recording.chunks.append(samples)
        except BaseException as error:
            recording.error = error
            self._last_error = f"{type(error).__name__}: {error}"
        finally:
            recording.ended.set()

    def _begin_recording(self) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("音频设备已关闭")
            if self._recording is not None:
                raise RuntimeError("已有录音正在进行")
            if self._playback is not None:
                self._stop_playback_locked(self._playback.identifier)
            stream = sd.InputStream(
                device=self.input_device, channels=1, samplerate=self.sample_rate, dtype="float32"
            )
            try:
                stream.start()
            except BaseException:
                stream.close(ignore_errors=False)
                raise
            recording = _Recording(identifier=uuid4().hex, stream=stream)
            recording.thread = Thread(target=self._record_loop, args=(recording,), daemon=False)
            self._recording = recording
            self._recordings[recording.identifier] = recording
            recording.thread.start()
            self._last_recording_outcome = None
            return recording.identifier

    async def begin_recording(self) -> str:
        return await asyncio.to_thread(self._begin_recording)

    def _finish_recording(self, recording_id: str) -> PayloadRef:
        with self._lock:
            recording = self._recording
            if recording is None or recording.identifier != recording_id:
                raise ValueError("recording_id 不是当前录音")
            recording.stop_requested.set()
            assert recording.thread is not None
            recording.thread.join(timeout=2)
            if recording.thread.is_alive():
                recording.stream.abort(ignore_errors=False)
                recording.thread.join(timeout=2)
            if recording.thread.is_alive():
                raise RuntimeError("录音读取线程未能停止")
            stop_error: BaseException | None = None
            try:
                recording.stream.stop(ignore_errors=False)
            except BaseException as error:
                stop_error = error
            try:
                recording.stream.close(ignore_errors=False)
            except BaseException as error:
                if stop_error is None:
                    stop_error = error
            self._recording = None
            if recording.error is not None:
                self._last_recording_outcome = "failed"
                raise RuntimeError("录音读取失败") from recording.error
            if stop_error is not None:
                self._last_recording_outcome = "failed"
                self._last_error = f"{type(stop_error).__name__}: {stop_error}"
                raise RuntimeError("录音流停止或关闭失败") from stop_error
            if not recording.chunks:
                self._last_recording_outcome = "failed"
                raise RuntimeError("录音没有音频帧")
            samples = np.concatenate(recording.chunks)
            path = self.output_dir / f"{recording.identifier}.wav"
            with path.open("xb") as output:
                sf.write(output, samples, self.sample_rate, format="WAV", subtype="PCM_16")
            _validate_wav(path)
            self._last_recording_outcome = "completed"
            recording.chunks.clear()
            return PayloadRef(path.as_uri(), "audio/wav", _sha256(path))

    async def finish_recording(self, recording_id: str) -> PayloadRef:
        return await asyncio.to_thread(self._finish_recording, recording_id)

    def _wait_recording_error(self, recording_id: str) -> str | None:
        with self._lock:
            recording = self._recordings.get(recording_id)
            if recording is None:
                raise ValueError("recording_id 不是当前录音")
        recording.ended.wait()
        with self._lock:
            self._recordings.pop(recording_id, None)
        return None if recording.error is None else f"{type(recording.error).__name__}: {recording.error}"

    async def wait_recording_error(self, recording_id: str) -> str | None:
        return await asyncio.to_thread(self._wait_recording_error, recording_id)

    def _play(self, audio: PayloadRef) -> str:
        path = _source_path(audio.uri)
        _validate_wav(path)
        if audio.sha256 is not None and _sha256(path) != audio.sha256:
            raise ValueError("播放文件 SHA256 不匹配")
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if samples.shape[1] != 1:
            raise ValueError("目前仅支持单声道播放")
        sd.check_output_settings(device=self.output_device, channels=1, samplerate=sample_rate, dtype="float32")
        with self._lock:
            if self._closed:
                raise RuntimeError("音频设备已关闭")
            if self._recording is not None:
                raise RuntimeError("录音期间不能播放")
            if self._playback is not None:
                self._stop_playback_locked(self._playback.identifier)
            playback_id = uuid4().hex
            playback: _Playback

            def callback(outdata, frames, _time, status) -> None:
                if status:
                    playback.error = RuntimeError(f"播放 callback 状态: {status}")
                    self._last_error = str(playback.error)
                    outdata.fill(0)
                    raise sd.CallbackAbort
                end = min(playback.position + frames, len(playback.samples))
                copied = end - playback.position
                outdata[:copied] = playback.samples[playback.position:end]
                if copied < frames:
                    outdata[copied:].fill(0)
                playback.position = end
                if end == len(playback.samples):
                    raise sd.CallbackStop

            def finished_callback() -> None:
                playback.outcome = (
                    "failed" if playback.error is not None else
                    "stopped" if playback.stop_requested.is_set() else "completed"
                )
                playback.finished.set()

            stream = sd.OutputStream(
                device=self.output_device, channels=1, samplerate=sample_rate, dtype="float32",
                callback=callback, finished_callback=finished_callback,
            )
            playback = _Playback(playback_id, stream, np.ascontiguousarray(samples))
            try:
                stream.start()
            except BaseException:
                stream.close(ignore_errors=False)
                raise
            self._playback = playback
            self._playbacks[playback.identifier] = playback
            return playback.identifier

    async def play(self, audio: PayloadRef) -> str:
        return await asyncio.to_thread(self._play, audio)

    @staticmethod
    def _close_playback_stream(playback: _Playback) -> None:
        with playback.stream_lock:
            if not playback.closed:
                playback.stream.close(ignore_errors=False)
                playback.closed = True
                playback.samples = np.empty((0, 1), dtype="float32")

    def _stop_playback_locked(self, playback_id: str) -> str:
        playback = self._playback
        if playback is None or playback.identifier != playback_id:
            raise ValueError("playback_id 不是当前播放")
        with playback.stream_lock:
            if not playback.finished.is_set():
                playback.stop_requested.set()
                playback.stream.abort(ignore_errors=False)
        if not playback.finished.wait(timeout=2):
            raise RuntimeError("播放流未能停止")
        self._close_playback_stream(playback)
        self._playback = None
        if playback.error is not None:
            raise RuntimeError("播放失败") from playback.error
        assert playback.outcome is not None
        return playback.outcome

    def _stop_playback(self, playback_id: str) -> str:
        with self._lock:
            return self._stop_playback_locked(playback_id)

    async def stop_playback(self, playback_id: str) -> str:
        return await asyncio.to_thread(self._stop_playback, playback_id)

    def _wait_playback(self, playback_id: str) -> str:
        with self._lock:
            playback = self._playbacks.get(playback_id)
            if playback is None:
                raise ValueError("playback_id 不是当前播放")
        playback.finished.wait()
        self._close_playback_stream(playback)
        with self._lock:
            self._playbacks.pop(playback_id, None)
        if playback.error is not None:
            raise RuntimeError("播放失败") from playback.error
        assert playback.outcome is not None
        return playback.outcome

    async def wait_playback(self, playback_id: str) -> str:
        return await asyncio.to_thread(self._wait_playback, playback_id)

    def status(self) -> dict:
        with self._lock:
            recording = self._recording
            playback = self._playback
            return {
                "recording_id": None if recording is None else recording.identifier,
                "recording_active": recording is not None and not recording.ended.is_set(),
                "recording_error": None if recording is None or recording.error is None else str(recording.error),
                "last_recording_outcome": self._last_recording_outcome,
                "playback_id": None if playback is None else playback.identifier,
                "playback_active": playback is not None and not playback.finished.is_set(),
                "playback_outcome": None if playback is None else playback.outcome,
                "playback_error": None if playback is None or playback.error is None else str(playback.error),
                "last_error": self._last_error,
                "closed": self._closed,
            }

    def _close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._playback is not None:
                self._stop_playback_locked(self._playback.identifier)
            if self._recording is not None:
                recording = self._recording
                recording.stop_requested.set()
                assert recording.thread is not None
                recording.thread.join(timeout=2)
                if recording.thread.is_alive():
                    recording.stream.abort(ignore_errors=False)
                    recording.thread.join(timeout=2)
                if recording.thread.is_alive():
                    raise RuntimeError("录音读取线程未能停止")
                recording.stream.close(ignore_errors=False)
                self._recording = None
            self._closed = True

    async def close(self) -> None:
        await asyncio.to_thread(self._close)
