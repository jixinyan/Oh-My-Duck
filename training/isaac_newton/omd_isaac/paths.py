"""Shared source and generated artifact locations without simulator imports."""
import hashlib
import json
import os
from pathlib import Path


def project_root():
    root = Path(os.environ.get("OMD_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
    if not (root / "configs/upstream.json").is_file():
        raise RuntimeError("Set OMD_PROJECT_ROOT to the project checkout")
    return root


def asset_source():
    return project_root() / ".cache/upstream/microduck_rl/src/mjlab_microduck/robot/microduck"


ROBOT_MODELS = {"walk": "robot_walk.xml", "groundcontact": "robot_groundcontact.xml"}


def model_file(model="walk"):
    if model not in ROBOT_MODELS:
        raise ValueError(f"Unknown robot model {model!r}; choose from {tuple(ROBOT_MODELS)}")
    return ROBOT_MODELS[model]


def source_fingerprint(model="walk"):
    source = asset_source()
    if not source.is_dir():
        raise FileNotFoundError("Official robot sources missing; run setup first")
    h = hashlib.sha256()
    h.update(model_file(model).encode() + b"\0")
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            h.update(str(path.relative_to(source)).encode() + b"\0")
            h.update(hashlib.sha256(path.read_bytes()).digest())
    h.update((project_root() / "configs/upstream.json").read_bytes())
    package = Path(__file__).resolve().parent
    for tool in (package / "convert_asset.py", package / "asset_reference.py", package / "paths.py",
                 package.parent / "asset_converter/uv.lock"):
        h.update(tool.name.encode() + b"\0")
        h.update(hashlib.sha256(tool.read_bytes()).digest())
    return h.hexdigest()


def asset_dir(model="walk"):
    model_file(model)
    return project_root() / "artifacts/isaac-newton" / model / source_fingerprint(model)


def usd_path(model="walk"):
    stem = Path(model_file(model)).stem
    return asset_dir(model) / stem / (stem + ".usda")


def require_asset(model="walk"):
    manifest_path = asset_dir(model) / "build.json"
    if not manifest_path.is_file() or not usd_path(model).is_file():
        raise FileNotFoundError("Converted asset missing; submit: python omd.py assets --backend isaac-newton")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_fingerprint"] != source_fingerprint(model):
        raise RuntimeError("Asset source fingerprint differs from its build manifest")
    for relative, expected in manifest["files"].items():
        path = asset_dir(model) / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Generated asset changed or missing: {relative}")
    return usd_path(model)
