from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys

from oh_my_duck.agentic.skills.simulation import SimulationSkillRunner
from oh_my_duck.agentic.tools.simulation import simulation_tools
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend, IsaacNewtonBackend, MotionBusyError


async def _serve(args) -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
    def construct():
        if args.backend == "cpu-mujoco-bam":
            return CpuMujocoBamBackend(robot_id=args.robot_id, policy_path=args.policy, task_id=args.task)
        return IsaacNewtonBackend(robot_id=args.robot_id, policy_path=args.policy, task_id=args.task, device=args.device)
    backend = await asyncio.get_running_loop().run_in_executor(executor, construct)
    backend.bind_executor(executor)
    runner = SimulationSkillRunner(backend)
    catalog = simulation_tools(backend, runner)
    output_lock = asyncio.Lock()
    pending: set[asyncio.Task] = set()
    fatal = asyncio.get_running_loop().create_future()

    async def publish(message: dict) -> None:
        async with output_lock:
            print(json.dumps(message, allow_nan=False), file=sys.__stdout__, flush=True)

    async def invoke(message: dict) -> None:
        request_id = message.get("request_id")
        try:
            if set(message) != {"tool", "request_id", "arguments"} or not isinstance(message["tool"], str) or not isinstance(request_id, str) or not request_id or not isinstance(message["arguments"], dict):
                raise ValueError("Each line requires tool, nonempty request_id and object arguments")
            result = await catalog.invoke(message["tool"], request_id, message["arguments"])
            await publish(asdict(result))
        except (ValueError, MotionBusyError) as error:
            await publish({"request_id": request_id, "status": "rejected", "payload": {"reason": str(error)}})

    async def read_lines() -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                return
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("Request line must be a JSON object")
            except (json.JSONDecodeError, ValueError) as error:
                await publish({"request_id": None, "status": "rejected", "payload": {"reason": str(error)}})
                continue
            task = asyncio.create_task(invoke(message))
            pending.add(task)
            task.add_done_callback(pending.discard)
            task.add_done_callback(lambda completed: fatal.set_exception(completed.exception()) if completed.exception() is not None and not fatal.done() else None)

    ticker = asyncio.create_task(backend.run())
    reader = asyncio.create_task(read_lines())
    try:
        done, _ = await asyncio.wait((ticker, reader, fatal), return_when=asyncio.FIRST_COMPLETED)
        for item in done:
            await item
        if pending:
            await asyncio.gather(*pending)
    finally:
        ticker.cancel()
        reader.cancel()
        await asyncio.gather(ticker, reader, return_exceptions=True)
        await asyncio.get_running_loop().run_in_executor(executor, backend.close)
        executor.shutdown(wait=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive measured simulation, JSONL requests on stdin and JSONL results on stdout")
    parser.add_argument("--backend", required=True, choices=("cpu-mujoco-bam", "isaac-newton"))
    parser.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--robot-id", default="sim-microduck-1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        asyncio.run(_serve(args))
    return 0
