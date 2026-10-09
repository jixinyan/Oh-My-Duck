import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

from oh_my_duck.integrations.native_client import NativeTaskClient, validate_context_run_ids


RUN_ID = "278aab2a-42a1-41df-a0e1-7eac0421f81b"


@pytest.mark.parametrize("context", [None, RUN_ID, {}, {RUN_ID}, [RUN_ID, RUN_ID],
                                    [RUN_ID] * 5, [""], ["a" * 81], ["../run"], ["任务"],
                                    ["run\n"], [1], [True], [[]]])
def test_invalid_context_fails_before_native_session_or_http(tmp_path, context):
    async def run():
        client = NativeTaskClient("http://127.0.0.1:1", "profile", "scenario", tmp_path / "native")
        try:
            with pytest.raises(ValueError, match="context_run_ids"):
                await client.submit("读取当前相机。", context_run_ids=context)
            assert client.session_id is None and client.run_id is None
            assert list(client.output.iterdir()) == []
        finally:
            await client.close()
        assert client._client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("instruction", [None, True, 42, [], {}, "", " \n", "a" * 4001, "🦆" * 2001])
def test_invalid_instruction_fails_before_native_session_or_http(tmp_path, instruction):
    async def run():
        client = NativeTaskClient("http://127.0.0.1:1", "profile", "scenario", tmp_path / "native")
        try:
            with pytest.raises(ValueError, match="UTF-16"):
                await client.submit(instruction)
            assert list(client.output.iterdir()) == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("instruction", ["读取当前相机。", "a" * 4000, "🦆" * 2000])
def test_native_instruction_limit_accepts_utf16_boundary(tmp_path, instruction):
    async def run():
        client = NativeTaskClient("http://127.0.0.1:1", "profile", "scenario", tmp_path / "native")
        try:
            with pytest.raises(RuntimeError, match="必须打开"):
                await client.submit(instruction, context_run_ids=(RUN_ID,))
            assert list(client.output.iterdir()) == []
        finally:
            await client.close()

    asyncio.run(run())


def test_context_validation_preserves_selection_and_copies_input():
    selected = [RUN_ID]
    admitted = validate_context_run_ids(selected)
    selected.clear()
    assert admitted == (RUN_ID,)
    assert validate_context_run_ids(()) == ()


def test_previous_context_requires_multiple_audio_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "voice"
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-m", "oh_my_duck", "voice-task", "--audio",
                             str(tmp_path / "unavailable.wav"), "--persona", "duck", "--harness-url",
                             "http://127.0.0.1:1", "--profile", "profile", "--scenario", "scenario",
                             "--output", str(output), "--context-previous-task"], cwd=Path.home(),
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "需要至少两项录音" in result.stderr
    assert not output.exists()
