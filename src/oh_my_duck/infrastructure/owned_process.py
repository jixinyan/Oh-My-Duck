from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class OwnedProcess:
    def __init__(self, command, *, record_path: Path, cancel_stdin: bool = False, **kwargs):
        self.record_path = record_path
        self.cancel_stdin = cancel_stdin
        if cancel_stdin and kwargs.get("stdin") != subprocess.PIPE:
            raise ValueError("Remote cancellation requires an owned input pipe")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("x") as record:
            record.write('{"state": "starting"}\n')
        self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
        self.record = {"pid": self.process.pid, "process_group_id": self.process.pid,
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "state": "running", "signals": [], "cancel_requested": False}
        self.save()

    def save(self):
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n")

    def send_signal(self, value):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, value)
            except ProcessLookupError:
                self.process.wait()
            self.record["signals"].append(signal.Signals(value).name)
            self.save()

    def stop(self, grace_s=120):
        if self.process.poll() is None:
            self.record["cancel_requested"] = True
            self.save()
            if self.cancel_stdin:
                self.process.stdin.close()
                self.record["control_channel_closed"] = True
                self.save()
            else:
                self.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self.record["grace_timeout"] = True
                self.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.send_signal(signal.SIGKILL)
                    self.process.wait(timeout=30)
        self.record.update(state="exited", exit_code=self.process.returncode,
                           ended_at=datetime.now(timezone.utc).isoformat())
        self.save()
        if self.record.get("grace_timeout"):
            raise TimeoutError("Owned process exceeded its graceful cleanup deadline; exit evidence is saved")


@contextmanager
def owned_process(command, *, record_path, cancel_stdin=False, **kwargs):
    owned = OwnedProcess(command, record_path=record_path, cancel_stdin=cancel_stdin, **kwargs)
    try:
        yield owned.process
    finally:
        failure = sys.exception()
        if failure is not None:
            owned.record["error_type"] = type(failure).__name__
        started = time.monotonic()
        owned.stop()
        owned.record["cleanup_elapsed_s"] = time.monotonic() - started
        owned.save()
        if owned.process.stdin is not None:
            owned.process.stdin.close()
