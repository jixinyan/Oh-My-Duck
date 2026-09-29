import asyncio
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_my_duck.agentic.harness import (
    DeterministicHarnessMock,
    HarnessEndpoint,
    HarnessUnavailable,
    TextRoute,
    UnavailableHarnessBridge,
)
from oh_my_duck.agentic.tools import ToolCatalog, ToolDefinition, ToolResult, ToolSchemaError
from oh_my_duck.core.contracts import EpisodeEvent, ExecutionDomain, Identity


class HarnessContractTests(unittest.TestCase):
    def test_schema_validation_happens_before_handler(self):
        catalog = ToolCatalog()
        called = []

        async def handler(request_id, arguments):
            called.append(arguments)
            return ToolResult(request_id, "completed", {})

        catalog.register(
            ToolDefinition(
                "velocity",
                "Set a bounded velocity",
                {
                    "type": "object",
                    "required": ["vx"],
                    "additionalProperties": False,
                    "properties": {"vx": {"type": "number", "minimum": -1, "maximum": 1}},
                },
            ),
            handler,
        )
        with self.assertRaises(ToolSchemaError):
            asyncio.run(catalog.invoke("velocity", "r1", {"vx": 2}))
        self.assertEqual(called, [])

    def test_mock_routes_only_registered_tools_and_is_idempotent(self):
        catalog = ToolCatalog()
        calls = []

        async def handler(request_id, arguments):
            calls.append((request_id, arguments))
            return ToolResult(request_id, "completed", {"accepted": True})

        definition = ToolDefinition("walk", "Walk at a bounded velocity", {"type": "object"})
        catalog.register(definition, handler)
        mock = DeterministicHarnessMock(catalog, {"向前走": TextRoute("walk", {"speed": 0.1})})
        asyncio.run(mock.register_tools((definition,)))

        result = asyncio.run(mock.submit_text("  向前走 ", session_id="s1", request_id="r1"))
        repeated = asyncio.run(mock.submit_text("向前走", session_id="s1", request_id="r1"))
        self.assertEqual(result.status, "completed")
        self.assertEqual(repeated, result)
        self.assertEqual(calls, [("r1", {"speed": 0.1})])

        unsupported = asyncio.run(mock.submit_text("跳舞", session_id="s1", request_id="r2"))
        self.assertEqual(unsupported.status, "unsupported")
        cancelled = asyncio.run(mock.cancel("r2", session_id="s1", reason="user stopped"))
        self.assertEqual(cancelled.status, "unsupported")

    def test_mock_preserves_event_and_experience_identity(self):
        mock = DeterministicHarnessMock(ToolCatalog())
        event = EpisodeEvent(
            1,
            "event-1",
            "episode-1",
            "session-1",
            Identity("duck", "sim-1", ExecutionDomain.SIMULATION),
            "task.finished",
            "2026-09-30T00:00:00Z",
            "simulation",
        )
        asyncio.run(mock.publish_event(event))
        asyncio.run(mock.publish_event(event))
        asyncio.run(mock.submit_experience(episode_id="episode-1", evidence_refs=("e1", "e1")))
        self.assertEqual(mock.events, (event,))
        self.assertEqual(mock.experiences, {"episode-1": ("e1",)})
        self.assertEqual(
            asyncio.run(mock.cancel("missing", session_id="s", reason="stop")).status,
            "unsupported",
        )

    def test_real_transport_is_explicitly_unavailable(self):
        bridge = UnavailableHarnessBridge(HarnessEndpoint("http://edh.invalid", source_revision="rc.2"))
        with self.assertRaises(HarnessUnavailable):
            asyncio.run(bridge.submit_text("向前走", session_id="s", request_id="r"))


if __name__ == "__main__":
    unittest.main()
