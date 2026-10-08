import asyncio

from oh_my_duck.integrations.edh.device import MicroDuckActionDevice
from oh_my_duck.integrations.edh.environment import MicroDuckEnvironment
from oh_my_duck.integrations.edh.session import MicroDuckWorkerSession
from oh_my_duck.integrations.edh.worker import serve

__all__ = ["MicroDuckActionDevice", "MicroDuckEnvironment", "MicroDuckWorkerSession", "serve"]


if __name__ == "__main__":
    asyncio.run(serve())
