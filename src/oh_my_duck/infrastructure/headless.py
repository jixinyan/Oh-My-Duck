"""Headless graphics configuration for the CUDA simulation environments."""

import os
from pathlib import Path


def configure_egl():
    """Respect explicit settings; prefer NVIDIA EGL on a CUDA host.

    Avoid implicit vendor selection on CUDA hosts. This alone does not guarantee
    valid frames; native readback is checked separately and OSMesa is explicit.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if os.environ["MUJOCO_GL"] == "egl" and vendor.is_file():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))


def rendering_environment(renderer):
    if renderer not in ("egl", "osmesa"):
        raise ValueError(f"Unknown MuJoCo renderer: {renderer}")
    configure_egl()
    environment = dict(os.environ, MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer)
    if renderer == "osmesa":
        from oh_my_duck.core.paths import project_root

        library = project_root() / ".cache/render-libs/osmesa/usr/lib/x86_64-linux-gnu"
        if library.is_dir():
            environment["LD_LIBRARY_PATH"] = str(library) + ":" + environment.get("LD_LIBRARY_PATH", "")
    return environment


def install_osmesa():
    """Install a checksum-pinned optional library locally; never change the OS."""
    import hashlib
    import json
    import platform
    import subprocess
    import urllib.request
    from oh_my_duck.core.paths import project_root

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "The pinned OSMesa archive targets Ubuntu 22.04 amd64; use system OSMesa on other hosts"
        )
    root = project_root()
    spec = json.loads((root / "configs/rendering.json").read_text())["osmesa_ubuntu_22_04_amd64"]
    cache = root / ".cache/render-libs"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "libosmesa6.deb"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        urllib.request.urlretrieve(spec["url"], temporary)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("OSMesa archive checksum mismatch")
    subprocess.run(["dpkg-deb", "-x", str(archive), str(cache / "osmesa")], check=True)
    library = cache / "osmesa/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"
    dependencies = subprocess.check_output(["ldd", str(library)], text=True)
    if "not found" in dependencies:
        raise RuntimeError("Missing system dependencies for OSMesa: " + dependencies)
    (cache / "provenance.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("Optional OSMesa installed in", cache / "osmesa")
