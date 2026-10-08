from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from physical_harness.execution.worker import require_object

from oh_my_duck.integrations.edh.session import MicroDuckWorkerSession


async def serve() -> None:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(3, "wb", buffering=0))
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    output_lock = asyncio.Lock()

    async def emit(message: dict[str, Any]) -> None:
        serialized = json.dumps(message, allow_nan=False, separators=(",", ":"))
        if len(serialized) > 32 * 1024 * 1024:
            raise ValueError("Native worker message exceeds the transport bound")
        async with output_lock:
            writer.write((serialized + "\n").encode())
            async with asyncio.timeout(10):
                await writer.drain()

    session = MicroDuckWorkerSession(emit)
    handlers = {
        "initialize": session.initialize,
        "open_task": session.open_task,
        "start": session.start,
        "pause": session.pause,
        "stop": lambda args: session.pause(args, terminal=True),
        "resume": session.resume,
        "capture": session.capture,
        "turn_view": session.turn_view,
        "check": session.check,
        "close_task": lambda _args: session.close_task(),
        "close": lambda _args: session.close(),
    }
    active: set[asyncio.Task[None]] = set()

    async def handle(message: dict[str, Any]) -> None:
        request_id = message["id"]
        try:
            operation = message["op"]
            if operation not in handlers:
                raise ValueError("Unknown native worker operation")
            result = await handlers[operation](require_object(message.get("args", {})))
            await emit({"id": request_id, "result": result})
        except Exception as error:
            await emit({"id": request_id, "error": {"type": type(error).__name__,
                                                   "message": str(error)}})

    while line := await asyncio.to_thread(sys.stdin.readline):
        if len(line) > 32 * 1024 * 1024:
            raise ValueError("Native worker request exceeds the transport bound")
        task = asyncio.create_task(handle(require_object(json.loads(line))))
        active.add(task)
        task.add_done_callback(active.discard)
    await session.transport_disconnected()
    for task in active:
        task.cancel()
    if active:
        await asyncio.gather(*active, return_exceptions=True)
