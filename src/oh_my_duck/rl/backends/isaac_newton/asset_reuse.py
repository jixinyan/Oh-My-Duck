import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.backends.isaac_newton.paths import asset_dir, require_asset, source_fingerprint


def _source_revision(root: Path) -> str:
    changes = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=no",
    ], cwd=root, text=True)
    if changes:
        raise ValueError("Asset reuse requires a clean committed source checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _conversion_inputs(root: Path) -> dict[str, str]:
    source = root / "src/oh_my_duck/robotics/microduck/microduck"
    paths = sorted(path for path in source.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix != ".pyc")
    if not paths:
        raise FileNotFoundError("Asset reuse requires the complete robot source directory")
    tools = root / "src/oh_my_duck/rl/backends/isaac_newton"
    paths.extend(tools / name for name in
                 ("convert_asset.py", "asset_reference.py", "paths.py", "asset_names.py"))
    paths.extend((root / "configs/upstream.json", root / "environments/isaac-assets/pyproject.toml"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _conversion_dependencies(root: Path) -> dict:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project.get("dependencies", []):
        raise ValueError("Project runtime dependencies require explicit asset conversion review")
    locked = tomllib.loads((root / "environments/isaac-assets/uv.lock").read_text())
    for package in locked["package"]:
        for dependency in package.get("dependencies", []):
            if dependency["name"] == "oh-my-duck" and dependency.get("extra"):
                raise ValueError("Asset reuse requires project optional dependencies to remain inactive")
    own = [package for package in locked["package"] if package["name"] == "oh-my-duck"]
    if len(own) != 1:
        raise ValueError("Asset conversion lock must contain exactly one project package")
    # 转换环境没有启用项目 extra；全部已解析依赖和 source 信息仍参与比较。
    own[0].pop("metadata")
    return locked


def _verify_files(directory: Path, manifest: dict) -> None:
    expected = manifest["files"]
    if not expected:
        raise ValueError("Asset build manifest must list its generated files")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Asset reuse requires generated files without symbolic links")
        if path.is_file():
            actual.add(str(path.relative_to(directory)))
    if actual != set(expected) | {"build.json"}:
        raise ValueError("Asset directory differs from the complete generated file manifest")
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset manifest file must remain within the generated directory")
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Generated asset file differs from its source manifest: {name}")


def reuse_asset(baseline_source: Path, model: str) -> dict:
    root = project_root()
    baseline = baseline_source.resolve(strict=True)
    current_revision, baseline_revision = _source_revision(root), _source_revision(baseline)
    target = asset_dir(model)
    if target.exists():
        raise FileExistsError(f"Asset reuse requires a new destination: {target}")
    inputs = _conversion_inputs(root)
    if inputs != _conversion_inputs(baseline):
        raise ValueError("Robot sources, conversion tools or conversion environment configuration differ")
    dependencies = _conversion_dependencies(root)
    if dependencies != _conversion_dependencies(baseline):
        raise ValueError("Resolved asset conversion dependencies differ")
    dependency_digest = hashlib.sha256(json.dumps(
        dependencies, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(baseline / "src"),
                   "OMD_PROJECT_ROOT": str(baseline), "TMPDIR": str(root / ".cache/tmp")}
    original_usd = Path(subprocess.check_output([
        sys.executable, "-c",
        "from oh_my_duck.rl.backends.isaac_newton.paths import require_asset; import sys; print(require_asset(sys.argv[1]))",
        model,
    ], cwd=baseline, env=environment, text=True).strip()).resolve(strict=True)
    original = original_usd.parent.parent
    original_bytes = (original / "build.json").read_bytes()
    manifest = json.loads(original_bytes)
    if manifest["model"] != model:
        raise ValueError("Verified baseline asset names another robot model")
    _verify_files(original, manifest)
    fingerprint = source_fingerprint(model)
    evidence = {
        "schema_version": 1,
        "baseline_source_revision": baseline_revision,
        "current_source_revision": current_revision,
        "baseline_source_fingerprint": manifest["source_fingerprint"],
        "current_source_fingerprint": fingerprint,
        "baseline_build_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "conversion_input_sha256": inputs,
        "conversion_dependency_sha256": dependency_digest,
        "generated_files_verified": len(manifest["files"]),
        "performed_conversion": False,
        "gpu_acceptance_performed": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reuse-", dir=target.parent) as temporary:
        work = Path(temporary)
        shutil.copytree(original, work, dirs_exist_ok=True)
        updated = copy.deepcopy(manifest)
        updated.update(source_fingerprint=fingerprint, source_reuse=evidence)
        (work / "build.json").write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n")
        _verify_files(work, updated)
        if (source_fingerprint(model) != fingerprint or _source_revision(root) != current_revision
                or _source_revision(baseline) != baseline_revision):
            raise RuntimeError("Asset conversion source changed during reuse verification")
        work.rename(target)
    verified = require_asset(model)
    return {"model": model, "usd_path": str(verified), "status": updated["status"],
            "source_reuse": evidence}
