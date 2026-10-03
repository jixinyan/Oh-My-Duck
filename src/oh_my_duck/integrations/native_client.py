import asyncio
import json
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx


TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "unknown"}


class NativeTaskClient:
    def __init__(self, url: str, profile_id: str, scenario: str, output: Path):
        endpoint = httpx.URL(url)
        if (endpoint.scheme != "http" or endpoint.host not in {"127.0.0.1", "localhost"}
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or endpoint.path not in {"", "/"}):
            raise ValueError("Harness 服务必须使用本机回环地址")
        if not profile_id.strip() or not scenario.strip():
            raise ValueError("profile_id 和 scenario 必须有内容")
        self.profile_id, self.scenario = profile_id, scenario
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self._client = httpx.AsyncClient(base_url=str(endpoint).rstrip("/"), trust_env=False,
            timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10))
        self.session_id = None
        self.run_id = None
        self._submission = asyncio.Lock()

    def _save(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
            allow_nan=False) + "\n", encoding="utf-8")

    async def _request(self, method, route, body=None):
        response = await self._client.request(method, route, **({} if body is None else {"json": body}))
        response.raise_for_status()
        return response.json()

    async def open(self):
        if self.session_id is not None:
            raise RuntimeError("Harness session 已经打开")
        record = await self._request("POST", "/api/sessions",
            {"profileId": self.profile_id, "requestId": str(uuid4())})
        self.session_id = record["id"]
        self._save("session.json", record)
        return record

    async def status(self):
        if self.run_id is None:
            raise RuntimeError("尚未提交 Harness task")
        run = await self._request("GET", f"/api/runs/{quote(self.run_id, safe='')}?events=none")
        if run["id"] != self.run_id or run["source"] not in {"simulation", "hardware"}:
            raise ValueError("Harness task 身份或来源不一致")
        self._save(f"{self.run_id}-status.json", run)
        return run

    async def submit(self, instruction: str):
        if not instruction.strip():
            raise ValueError("语音指令必须有内容")
        async with self._submission:
            if self.session_id is None:
                raise RuntimeError("必须打开 Harness session")
            if self.run_id is not None and (await self.status())["state"] not in TERMINAL_STATES:
                raise RuntimeError("已有任务正在执行；请中断任务后提交新指令")
            route = f"/api/sessions/{quote(self.session_id, safe='')}/tasks"
            catalog = await self._request("GET", route)
            if self.scenario not in catalog["tasks"]:
                raise ValueError("scenario 不在当前原生任务目录中")
            submission = {"scenario": self.scenario, "requestId": str(uuid4()),
                "catalogRevision": catalog["descriptor"]["digest"], "instruction": instruction}
            self._save(f"{submission['requestId']}-submission.json", submission)
            result = await self._request("POST", route, submission)
            self.run_id = result["runId"]
            self._save(f"{self.run_id}-submission-result.json", result)
            return result

    async def wait(self, *, timeout_s=2400):
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
                self.session_id = None
        finally:
            await self._client.aclose()
