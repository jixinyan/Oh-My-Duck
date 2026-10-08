from __future__ import annotations

from typing import Any, Callable

from physical_harness.environments import NativeObservation
from physical_harness.execution.native_device import NativeActionDevice

from oh_my_duck.integrations.edh.environment import MicroDuckEnvironment


class MicroDuckActionDevice(NativeActionDevice):
    def __init__(self, environment: MicroDuckEnvironment,
                 publish_pausing: Callable[[NativeObservation], Any]) -> None:
        super().__init__(environment)
        self._publish_pausing = publish_pausing

    async def stop(self, execution_id: str, generation: int) -> dict:
        acknowledgement = await super().stop(execution_id, generation)
        observation = await self.on_owner(self.environment.observe)
        await self._publish_pausing(observation)
        return acknowledgement
