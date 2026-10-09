import asyncio
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
from uuid import uuid4

import httpx


TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "unknown"}


def validate_context_run_ids(value: object) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) > 4
            or any(not isinstance(item, str) or re.fullmatch(r"[A-Za-z0-9-]{1,80}", item) is None
                   for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("context_run_ids 必须包含最多四个不同的原生任务编号")
    return tuple(value)


class NativeTaskClient:
    def __init__(self, url: str, profile_id: str, scenario: str, output: Path,
                 expected_source: str | None = None):
        endpoint = httpx.URL(url)
        if (endpoint.scheme != "http" or endpoint.host not in {"127.0.0.1", "localhost"}
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or endpoint.path not in {"", "/"}):
            raise ValueError("Harness 服务必须使用本机回环地址")
        if not profile_id.strip() or not scenario.strip():
            raise ValueError("profile_id 和 scenario 必须有内容")
        self.profile_id, self.scenario = profile_id, scenario
        if expected_source not in {None, "simulation", "hardware"}:
            raise ValueError("expected_source 必须是 simulation 或 hardware")
        self.expected_source = expected_source
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self._client = httpx.AsyncClient(base_url=str(endpoint).rstrip("/"), trust_env=False,
            timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10))
        self.session_id = None
        self.run_id = None
        self._source = None
        self._submission = asyncio.Lock()

    def _save(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
            allow_nan=False) + "\n", encoding="utf-8")

    async def _request(self, method, route, body=None):
        response = await self._client.request(method, route, **({} if body is None else {"json": body}))
        if response.is_error:
            self._save(f"http-{uuid4().hex}-error.json", {"method": method, "route": route,
                "status": response.status_code, "response": response.json()})
        response.raise_for_status()
        return response.json()

    async def open(self):
        if self.session_id is not None:
            raise RuntimeError("Harness session 已经打开")
        record = await self._request("POST", "/api/sessions",
            {"profileId": self.profile_id, "requestId": str(uuid4())})
        self.session_id = record["id"]
        self._save("session.json", record)
        source = record["configuration"]["mode"]
        if (record["profileId"] != self.profile_id or record["state"] != "ready"
                or record["resources"] != "held" or source not in {"simulation", "hardware"}
                or self.expected_source not in {None, source}):
            raise ValueError("Harness session 与所选执行领域不一致")
        self._source = source
        return record

    async def status(self):
        if self.run_id is None:
            raise RuntimeError("尚未提交 Harness task")
        run = await self._request("GET", f"/api/runs/{quote(self.run_id, safe='')}?events=none")
        if run["id"] != self.run_id or run["source"] != self._source:
            raise ValueError("Harness task 身份或来源不一致")
        self._save(f"{self.run_id}-status.json", run)
        return run

    async def _wait_ready(self):
        async with asyncio.timeout(240):
            while True:
                record = await self._request("GET", f"/api/sessions/{quote(self.session_id, safe='')}")
                self._save("session-status.json", record)
                if (record["id"] != self.session_id or record["profileId"] != self.profile_id
                        or record["configuration"]["mode"] != self._source
                        or record["resources"] != "held"):
                    raise ValueError("Harness session 身份、来源或资源状态不一致")
                if record["state"] == "ready":
                    return record
                if (record["state"] not in {"running", "draining"} or self.run_id is None
                        or record["taskHistory"]["lastRunId"] != self.run_id):
                    raise RuntimeError("Harness session 无法接受下一项任务")
                await asyncio.sleep(0.2)

    async def submit(self, instruction: str, *, context_run_ids: tuple[str, ...] = ()):
        if (not isinstance(instruction, str) or not instruction.strip()
                or len(instruction.encode("utf-16-le")) > 8000):
            raise ValueError("语音指令必须包含 1–4000 个 UTF-16 code unit")
        context = validate_context_run_ids(context_run_ids)
        async with self._submission:
            if self.session_id is None:
                raise RuntimeError("必须打开 Harness session")
            if self.run_id is not None and (await self.status())["state"] not in TERMINAL_STATES:
                raise RuntimeError("已有任务正在执行；请中断任务后提交新指令")
            await self._wait_ready()
            route = f"/api/sessions/{quote(self.session_id, safe='')}/tasks"
            catalog = await self._request("GET", route)
            if self.scenario not in catalog["tasks"]:
                raise ValueError("scenario 不在当前原生任务目录中")
            submission = {"scenario": self.scenario, "requestId": str(uuid4()),
                "catalogRevision": catalog["descriptor"]["digest"], "instruction": instruction,
                "contextRunIds": list(context)}
            self._save(f"{submission['requestId']}-submission.json", submission)
            result = await self._request("POST", route, submission)
            self.run_id = result["runId"]
            self._save(f"{self.run_id}-submission-result.json", result)
            return result

    async def wait(self, *, timeout_s=2400):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s 必须是有限的正数")
        async with asyncio.timeout(timeout_s):
            while True:
                run = await self.status()
                if run["state"] in TERMINAL_STATES:
                    return run
                await asyncio.sleep(1)

    async def stop(self):
        async with self._submission:
            if self.run_id is None:
                return {"run_id": None, "state": "idle", "executions": []}
            run = await self.status()
            if run["state"] not in TERMINAL_STATES:
                accepted = await self._request("POST", f"/api/runs/{quote(self.run_id, safe='')}/stop", {})
                self._save(f"{self.run_id}-stop.json", accepted)
                run = await self.wait(timeout_s=240)
            if any(item["state"] != "ended" or not item["device_confirmed"] for item in run["executions"]):
                raise RuntimeError("原生 execution 尚未确认终止")
            return {"run_id": self.run_id, "state": run["state"], "executions": run["executions"]}

    async def close(self):
        try:
            if self.session_id is not None:
                record = await self._request("POST", f"/api/sessions/{quote(self.session_id, safe='')}/close", {})
                self._save("session-closed.json", record)
                if (record["id"] != self.session_id or record["profileId"] != self.profile_id
                        or record["state"] != "closed" or record["resources"] != "released"):
                    raise RuntimeError("Harness session 尚未确认关闭并释放资源")
                self.session_id = None
        finally:
            await self._client.aclose()
