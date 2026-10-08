from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HarnessBridge(Protocol):
    # 原生任务服务管理会话、规划、工具注册与经验资料。
    profile_id: str
    scenario: str
    session_id: str | None
    run_id: str | None

    async def open(self) -> dict[str, Any]: ...

    async def status(self) -> dict[str, Any]: ...

    async def submit(self, instruction: str) -> dict[str, Any]: ...

    async def wait(self, *, timeout_s: float = 2400) -> dict[str, Any]: ...

    async def stop(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...
