import asyncio
from pathlib import Path
import tempfile
import unittest

from oh_my_duck.agentic.application import ApplicationServices
from oh_my_duck.agentic.harness import HarnessBridge
from oh_my_duck.agentic.tools import ToolCatalog
from oh_my_duck.experience import JsonlEpisodeRecorder
from oh_my_duck.integrations.native_client import NativeTaskClient


class HarnessInterfaceTests(unittest.TestCase):
    def test_native_client_supplies_application_interface(self):
        temporary = Path.cwd() / ".cache/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            root = Path(directory)

            async def verify():
                client = NativeTaskClient(
                    "http://127.0.0.1:4318", "official-apartment-office", "navigate-office", root / "native",
                    expected_source="simulation",
                )
                try:
                    self.assertIsInstance(client, HarnessBridge)
                    services = ApplicationServices(ToolCatalog(), JsonlEpisodeRecorder(root / "episode.jsonl"),
                                                   harness=client)
                    self.assertIs(services.harness, client)
                    self.assertEqual(await services.harness.stop(),
                                     {"run_id": None, "state": "idle", "executions": []})
                    with self.assertRaises(RuntimeError):
                        await services.harness.status()
                    with self.assertRaises(ValueError):
                        await services.harness.submit(" ")
                finally:
                    await client.close()
                self.assertTrue(client._client.is_closed)

            asyncio.run(verify())

    def test_invalid_native_endpoint_fails_before_directory_creation(self):
        temporary = Path.cwd() / ".cache/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            output = Path(directory) / "native"
            for endpoint in ("https://127.0.0.1:4318", "http://example.com", "http://localhost/api",
                             "http://localhost?token=value", "http://user:password@localhost"):
                with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                    NativeTaskClient(endpoint, "profile", "scenario", output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
