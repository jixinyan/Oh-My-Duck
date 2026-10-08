from __future__ import annotations

import asyncio
import json
import socket
from uuid import uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oh_my_duck.integrations.edh_native import MicroDuckWorkerSession


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def control(port: int, secret: str, run_task_id: str, operation: str,
                  arguments: dict) -> dict:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        request = {"run_task_id": run_task_id, "control_secret": secret,
                   "operation": operation, "request_id": uuid4().hex,
                   "arguments": arguments}
        writer.write((json.dumps(request, allow_nan=False) + "\n").encode())
        await writer.drain()
        response = json.loads(await reader.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]
    finally:
        writer.close()
        await writer.wait_closed()


async def expect_control_error(port: int, secret: str, run_task_id: str,
                               operation: str, arguments: dict,
                               message: str) -> None:
    try:
        await control(port, secret, run_task_id, operation, arguments)
    except RuntimeError as error:
        if message not in str(error):
            raise
    else:
        raise AssertionError(f"Native control accepted a request requiring {message}")


async def wait_boundary(session: MicroDuckWorkerSession, *, timeout_s: float | None = 180) -> dict:
    pump = session._pump
    if pump is None:
        raise RuntimeError("Native policy pump was not started")
    async with asyncio.timeout(timeout_s):
        await asyncio.shield(pump)
        await session._await_motion_cleanup()
    status = session._status
    if status["state"] != "paused" or not status["device_confirmed"]:
        raise AssertionError("Motion guard did not confirm a native paused boundary")
    return status
