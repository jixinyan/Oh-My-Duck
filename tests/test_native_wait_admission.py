import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

from oh_my_duck.integrations.native_client import NativeTaskClient


@pytest.mark.parametrize("timeout", [None, True, False, "1", [], {}, 0, -1,
                                   float("nan"), float("inf"), -float("inf")])
def test_invalid_wait_timeout_fails_before_status_request(tmp_path, timeout):
    async def run():
        client = NativeTaskClient("http://127.0.0.1:1", "profile", "scenario", tmp_path / "native")
        try:
            with pytest.raises(ValueError, match="timeout_s"):
                await client.wait(timeout_s=timeout)
            assert client.session_id is None and client.run_id is None
            assert list(client.output.iterdir()) == []
        finally:
            await client.close()
        assert client._client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0.1, 1, 2400])
def test_finite_positive_timeout_reaches_native_task_state_validation(tmp_path, timeout):
    async def run():
        client = NativeTaskClient("http://127.0.0.1:1", "profile", "scenario", tmp_path / "native")
        try:
            with pytest.raises(RuntimeError, match="尚未提交"):
                await client.wait(timeout_s=timeout)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "0", "-1"])
def test_voice_task_rejects_invalid_timeout_before_output_creation(tmp_path, timeout):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "voice"
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-m", "oh_my_duck", "voice-task", "--audio",
                             str(tmp_path / "unavailable.wav"), "--persona", "duck", "--harness-url",
                             "http://127.0.0.1:1", "--profile", "profile", "--scenario", "scenario",
                             "--output", str(output), "--timeout=" + timeout], cwd=Path.home(),
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "timeout 必须是有限的正数" in result.stderr
    assert not output.exists()
