import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_my_duck.core.contracts import EpisodeEvent, ExecutionDomain, Identity
from oh_my_duck.experience import JsonlEpisodeRecorder
from oh_my_duck.agentic.tools import ToolCatalog


class FrameworkTests(unittest.TestCase):
    def test_interfaces_import_without_simulators(self):
        for name in ("agentic.application", "robotics.backends", "core.contracts", "agentic.harness", "perception", "robotics.policies", "experience", "agentic.skills", "agentic.tools", "rl.training", "voice"):
            importlib.import_module("oh_my_duck." + name)
        for heavy in ("mujoco", "torch", "warp", "isaaclab"):
            self.assertNotIn(heavy, sys.modules)

    def test_newton_never_silently_falls_back(self):
        from oh_my_duck.rl.training.registry import BackendUnavailable, default_registry
        with self.assertRaises(BackendUnavailable):
            default_registry().get("isaac-newton", Path.cwd()).command("train", [])

    def test_unimplemented_tools_do_not_report_success(self):
        catalog = ToolCatalog()
        result = asyncio.run(catalog.invoke("run_skill", "request-1", {}))
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(catalog.definitions(), ())

    def test_tool_correlation_and_duplicate_registration(self):
        from oh_my_duck.agentic.skills.simulation import SimulationSkillRunner
        from oh_my_duck.agentic.tools.simulation import simulation_tools
        from oh_my_duck.robotics.backends.simulation import SimulationBackend

        backend = SimulationBackend(robot_id="capability-check", backend="cpu-mujoco-bam",
                                    policy_path=None, task_id="capability-check")
        catalog = simulation_tools(backend, SimulationSkillRunner(backend))
        definition, handler = catalog._tools["get_capabilities"]
        with self.assertRaises(ValueError):
            catalog.register(definition, handler)
        result = asyncio.run(catalog.invoke("get_capabilities", "request-1", {}))
        self.assertEqual(result.request_id, "request-1")
        self.assertEqual(result.payload["identity"]["robot_id"], backend.robot_id)

    def test_episode_domain_and_evidence_survive_recording(self):
        temporary = Path.cwd() / ".cache/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            path = Path(directory) / "episode.jsonl"
            recorder = JsonlEpisodeRecorder(path)
            for i, domain in enumerate((ExecutionDomain.SIMULATION, ExecutionDomain.REAL)):
                recorder.append(EpisodeEvent(
                    1, f"event-{i}", f"episode-{i}", "session",
                    Identity("persona", f"robot-{i}", domain),
                    "task.finished", "2026-09-06T00:00:00Z", "host-utc",
                    {"status": "cancelled"}, evidence_refs=(f"evidence-{i}",)))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([e["identity"]["domain"] for e in events], ["simulation", "real"])
            self.assertEqual(events[0]["payload"]["status"], "cancelled")
            self.assertEqual(events[1]["evidence_refs"], ["evidence-1"])


if __name__ == "__main__":
    unittest.main()
