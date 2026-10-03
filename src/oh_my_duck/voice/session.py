import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from oh_my_duck.core.contracts.events import EpisodeEvent
from oh_my_duck.core.contracts.identity import ExecutionDomain, Identity
from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.experience.jsonl import JsonlEpisodeRecorder
from oh_my_duck.integrations.native_client import NativeTaskClient
from oh_my_duck.voice.base import AudioDevice
from oh_my_duck.voice.remote import RemoteVoiceServices


class VoiceSession:
    def __init__(
        self, device: AudioDevice, services: RemoteVoiceServices, *, persona_id: str,
        robot_id: str, domain: ExecutionDomain, log_path: Path | None,
        tasks: NativeTaskClient | None = None,
    ):
        self.device = device
        self.services = services
        self.tasks = tasks
        self.identity = Identity(persona_id, robot_id, domain)
        self.session_id = uuid4().hex
        self.episode_id = uuid4().hex
        self.recorder = None if log_path is None else JsonlEpisodeRecorder(log_path)
        self.last_audio: PayloadRef | None = None
        self._generation = 0
        self._transition = asyncio.Lock()
        self._pending: dict[str, asyncio.Task] = {}
        self._speech_requests: set[str] = set()
        self._watchers: set[asyncio.Task] = set()
        self._fatal: asyncio.Future | None = None

    def emit(self, kind: str, payload: dict, request_id: str | None = None) -> None:
        event = EpisodeEvent(
            schema_version=1, event_id=uuid4().hex, episode_id=self.episode_id,
            session_id=self.session_id, identity=self.identity, kind=kind,
            occurred_at=datetime.now(timezone.utc).isoformat(), clock_domain="host-utc",
            payload=payload, request_id=request_id,
        )
        if self.recorder is not None:
            self.recorder.append(event)
        print(json.dumps(asdict(event), ensure_ascii=False, allow_nan=False), flush=True)

    def _cancel_speech(self) -> None:
        self._generation += 1
        for request_id in tuple(self._speech_requests):
            self._pending[request_id].cancel()

    def _track(self, task: asyncio.Task) -> None:
        self._watchers.add(task)

        def completed(done: asyncio.Task) -> None:
            self._watchers.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None and self._fatal is not None and not self._fatal.done():
                    self._fatal.set_exception(error)

        task.add_done_callback(completed)

    async def _transcribe(self, request_id: str, audio: PayloadRef, language_hint: str | None,
                          execute: bool, generation: int) -> None:
        result = await self.services.transcribe(audio, language_hint=language_hint)
        self.emit("voice.transcription.completed", asdict(result), request_id)
        if execute:
            async with self._transition:
                if generation != self._generation:
                    self.emit("voice.task.discarded", {"text": result.text}, request_id)
                    return
                submitted = await self.tasks.submit(result.text)
                self.emit("voice.task.started", {**submitted, "transcription": asdict(result)}, request_id)
            run = await self.tasks.wait()
            self.emit("voice.task.ended", {"run_id": run["id"], "state": run["state"],
                "verdicts": run["verdicts"], "error": run["error"]}, request_id)
            if generation != self._generation:
                return
            text = {"succeeded": "任务已经完成。", "failed": "任务执行失败，请查看任务记录。",
                "cancelled": "任务已经中断。", "interrupted": "任务已经中断。",
                "unknown": "任务状态无法确认，请查看任务记录。"}[run["state"]]
            await self._speak(request_id, text, generation)

    async def _watch_playback(self, playback_id: str, request_id: str, metadata: dict) -> None:
        try:
            outcome = await self.device.wait_playback(playback_id)
            if outcome == "completed":
                self.emit("voice.playback.completed", {"playback_id": playback_id, **metadata}, request_id)
        except Exception as error:
            self.emit(
                "voice.playback.error",
                {"playback_id": playback_id, "error_type": type(error).__name__, "message": str(error)},
                request_id,
            )
            raise

    async def _watch_recording(self, recording_id: str, request_id: str) -> None:
        error = await self.device.wait_recording_error(recording_id)
        if error is not None:
            self.emit("voice.recording.error", {"recording_id": recording_id, "message": error}, request_id)
            raise RuntimeError(error)

    async def _speak(self, request_id: str, text: str, generation: int) -> None:
        speech = await self.services.synthesize(self.identity.persona_id, text)
        metadata = {
            "audio": speech.audio.uri, "audio_sha256": speech.audio.sha256,
            "persona_id": speech.persona_id, "voice_id": speech.voice_id,
            "profile_revision": speech.profile_revision, "model_revision": speech.model_revision,
        }
        async with self._transition:
            if generation != self._generation:
                self.emit("voice.speech.discarded", metadata, request_id)
                return
            if self.device.status()["recording_id"] is not None:
                raise RuntimeError("录音期间不能播放")
            playback_id = await self.device.play(speech.audio)
            self.emit("voice.playback.started", {"playback_id": playback_id, **metadata}, request_id)
            self._track(asyncio.create_task(
                self._watch_playback(playback_id, request_id, metadata)
            ))

    async def _run_request(self, request_id: str, command: str, operation) -> None:
        try:
            await operation
        except asyncio.CancelledError:
            self.emit("voice.request.cancelled", {"command": command}, request_id)
        except Exception as error:
            self.emit(
                "voice.request.error",
                {"command": command, "error_type": type(error).__name__, "message": str(error)},
                request_id,
            )
            raise
        finally:
            self._pending.pop(request_id, None)
            self._speech_requests.discard(request_id)

    def _start_request(self, request_id: str, command: str, operation) -> None:
        if request_id in self._pending:
            raise ValueError("request_id 已在执行")
        task = asyncio.create_task(self._run_request(request_id, command, operation))
        self._pending[request_id] = task
        if command == "speak":
            self._speech_requests.add(request_id)
        self._track(task)
        self.emit("voice.request.accepted", {"command": command}, request_id)

    async def dispatch(self, message: dict) -> None:
        request_id = message["request_id"]
        command = message["command"]
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id 必须是非空字符串")
        if command == "status":
            self.emit(
                "voice.status", {**self.device.status(), "pending_requests": list(self._pending)},
                request_id,
            )
        elif command == "begin_recording":
            async with self._transition:
                self._cancel_speech()
                recording_id = await self.device.begin_recording()
                self.emit("voice.recording.started", {"recording_id": recording_id}, request_id)
                self._track(asyncio.create_task(
                    self._watch_recording(recording_id, request_id)
                ))
        elif command == "finish_recording":
            recording_id = message["recording_id"]
            self.last_audio = await self.device.finish_recording(recording_id)
            self.emit(
                "voice.recording.finished",
                {"recording_id": recording_id, "audio": self.last_audio.uri,
                 "audio_sha256": self.last_audio.sha256}, request_id,
            )
        elif command in {"transcribe", "execute_recording"}:
            if command == "execute_recording" and self.tasks is None:
                raise RuntimeError("execute_recording 需要配置原生 Harness session")
            if "audio" in message:
                audio = PayloadRef(Path(message["audio"]).expanduser().resolve(strict=True).as_uri(), "audio/wav")
            else:
                if self.last_audio is None:
                    raise ValueError("尚无已完成录音")
                audio = self.last_audio
            self._start_request(
                request_id, command,
                self._transcribe(request_id, audio, message.get("language_hint"),
                    command == "execute_recording", self._generation),
            )
        elif command == "speak":
            if self.device.status()["recording_id"] is not None:
                raise RuntimeError("录音期间不能播放")
            async with self._transition:
                self._cancel_speech()
                generation = self._generation
                self._start_request(
                    request_id, command, self._speak(request_id, message["text"], generation)
                )
        elif command == "stop":
            async with self._transition:
                self._cancel_speech()
                for pending_id, task in tuple(self._pending.items()):
                    if pending_id not in self._speech_requests:
                        task.cancel()
                state = self.device.status()
                stopped: dict = {}
                if state["playback_id"] is not None:
                    outcome = await self.device.stop_playback(state["playback_id"])
                    stopped.update({"playback_id": state["playback_id"], "playback_outcome": outcome})
                if state["recording_id"] is not None:
                    self.last_audio = await self.device.finish_recording(state["recording_id"])
                    stopped.update({
                        "recording_id": state["recording_id"],
                        "audio": self.last_audio.uri, "audio_sha256": self.last_audio.sha256,
                    })
                if self.tasks is not None:
                    robot = await self.tasks.stop()
                    stopped["robot"] = robot
                    self.emit("voice.task.stop_confirmed", robot, request_id)
                self.emit("voice.stopped", stopped, request_id)
        else:
            raise ValueError(f"未知命令: {command}")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._fatal = loop.create_future()
        reader = asyncio.StreamReader()
        transport = None
        try:
            if self.tasks is not None:
                await self.tasks.open()
            transport, _protocol = await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
            )
            self.emit("voice.session.started", {"audio_device": self.device.status()})
            while True:
                line_task = asyncio.create_task(reader.readline())
                done, _pending = await asyncio.wait(
                    {line_task, self._fatal}, return_when=asyncio.FIRST_COMPLETED
                )
                if self._fatal in done:
                    line_task.cancel()
                    await self._fatal
                line = line_task.result()
                if not line:
                    break
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("JSONL 命令必须是 object")
                try:
                    await self.dispatch(message)
                except Exception as error:
                    self.emit(
                        "voice.request.error",
                        {"command": message.get("command"), "error_type": type(error).__name__,
                         "message": str(error)}, message.get("request_id"),
                    )
                    raise
        finally:
            if transport is not None:
                transport.close()
            self._cancel_speech()
            tasks = tuple(self._pending.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.device.close()
                if self._watchers:
                    await asyncio.gather(*tuple(self._watchers), return_exceptions=True)
            finally:
                try:
                    if self.tasks is not None:
                        await self.tasks.close()
                finally:
                    await self.services.close()
            self.emit("voice.session.closed", self.device.status())
            if self._fatal.done():
                self._fatal.result()
