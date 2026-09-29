import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from oh_my_duck.infrastructure.headless import configure_egl


class HeadlessConfigurationTest(unittest.TestCase):
    def test_cuda_host_selects_vendor_before_child_launch(self):
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.is_file", return_value=True):
            configure_egl()
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
            self.assertTrue(os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"].endswith("10_nvidia.json"))

    def test_explicit_vendor_is_preserved(self):
        with (
            patch.dict(os.environ, {"__EGL_VENDOR_LIBRARY_FILENAMES": "/explicit/vendor.json"}, clear=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            configure_egl()
            self.assertEqual(os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"], "/explicit/vendor.json")

    def test_no_nvidia_installation_leaves_vendor_selection_alone(self):
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.is_file", return_value=False):
            configure_egl()
            self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", os.environ)

    def test_pinned_loader_is_prepended_for_child_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / ".cache/render-libs/osmesa/usr/lib/x86_64-linux-gnu"
            library.mkdir(parents=True)
            with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/system/lib"}, clear=True), \
                    patch("oh_my_duck.core.paths.project_root", return_value=root), \
                    patch("pathlib.Path.is_file", return_value=False):
                configure_egl()
                self.assertEqual(
                    os.environ["LD_LIBRARY_PATH"],
                    f"{library}:/system/lib",
                )
