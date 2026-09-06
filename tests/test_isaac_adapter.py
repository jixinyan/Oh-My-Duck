import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training/isaac_newton"))
from omd_isaac.contracts import JOINT_NAMES, joint_indices
from omd_isaac.paths import asset_dir, require_asset, source_fingerprint, usd_path
from oh_my_duck.training.registry import BackendUnavailable, default_registry
from oh_my_duck.training.isaac_newton import DIAGNOSTIC_TASK


class IsaacAdapterTests(unittest.TestCase):
    def test_reordered_and_passive_joints_preserve_policy_semantics(self):
        actual = ["passive_wheel", *reversed(JOINT_NAMES)]
        values = {name: i + 0.125 for i, name in enumerate(JOINT_NAMES)}
        actual_values = [-999, *[values[name] for name in reversed(JOINT_NAMES)]]
        restored = [actual_values[i] for i in joint_indices(actual)]
        self.assertEqual(restored, list(values.values()))
        with self.assertRaises(ValueError):
            joint_indices(actual[2:])
        with self.assertRaises(ValueError):
            joint_indices([*actual, JOINT_NAMES[0]])

    def test_backend_rejects_implicit_pd_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("isaac-newton", "isaac-assets"):
                executable = root / ".envs" / name / "bin/python"
                executable.parent.mkdir(parents=True)
                executable.touch()
            backend = default_registry().get("isaac-newton", root)
            with self.assertRaises(BackendUnavailable):
                backend.command("train", [])
            with self.assertRaises(BackendUnavailable):
                backend.command("train", ["--task", "Isaac-Other-v0"])
            train = backend.command("train", ["--task", DIAGNOSTIC_TASK])
            assets = backend.command("assets", [])
            self.assertIn("isaac-newton/bin/python", train.argv[0])
            self.assertIn("isaac-assets/bin/python", assets.argv[0])
            self.assertNotIn("isaaclab", sys.modules)

    def test_asset_startup_requires_explicit_acceptance_before_child_launch(self):
        from omd_isaac import assets
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(assets, "asset_dir", return_value=Path(directory)), \
             patch.object(sys, "prefix", directory), \
             patch.object(sys, "argv", ["assets"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(assets.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                assets.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_asset_acceptance_is_child_scoped_and_failure_is_preserved(self):
        from omd_isaac import assets
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(assets, "asset_dir", return_value=Path(directory)), \
             patch.object(sys, "prefix", directory), \
             patch.object(sys, "argv", ["assets", "--accept-eula"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(assets.subprocess, "run") as run:
            # Mock only: this test never imports Kit or accepts an actual license.
            run.return_value.returncode = 7
            self.assertEqual(assets.main(), 7)
            self.assertEqual(run.call_args.kwargs["env"]["OMNI_KIT_ACCEPT_EULA"], "YES")
            self.assertNotIn("OMNI_KIT_ACCEPT_EULA", os.environ)

    def test_generated_assets_require_complete_unchanged_content(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OMD_PROJECT_ROOT": directory}):
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/upstream.json").write_text("{}")
            source = root / "training/microduck/src/omd_microduck/robot/microduck"
            source.mkdir(parents=True)
            xml = source / "robot_walk.xml"
            xml.write_text("source A")
            with self.assertRaises(FileNotFoundError):
                require_asset()
            usd = usd_path()
            usd.parent.mkdir(parents=True)
            usd.write_text("USD fixture")
            manifest = {"source_fingerprint": source_fingerprint(), "files": {
                str(usd.relative_to(asset_dir())): hashlib.sha256(usd.read_bytes()).hexdigest()}}
            (asset_dir() / "build.json").write_text(json.dumps(manifest))
            self.assertEqual(require_asset(), usd)
            usd.write_text("changed output")
            with self.assertRaises(RuntimeError):
                require_asset()
            xml.write_text("source B")
            with self.assertRaises(FileNotFoundError):
                require_asset()

if __name__ == "__main__":
    unittest.main()
