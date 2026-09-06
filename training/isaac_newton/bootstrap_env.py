"""Prepare the isolated, pinned Isaac Lab / Newton environment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-env", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/upstream.json").read_text())
    spec = lock["isaac_candidate"]
    dest = ROOT / ".cache/upstream/IsaacLab"
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", spec["tag"], spec["url"], str(dest)], check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=dest, text=True).strip()
    if actual != spec["commit"] or dirty:
        raise SystemExit("Isaac Lab source differs from its clean pin; inspect before setup")
    subprocess.run([sys.executable, str(ROOT / "scripts/bootstrap.py"), "--skip-env"], check=True)
    if args.skip_env:
        return 0
    shared = {**os.environ, "UV_CACHE_DIR": str(ROOT / ".cache/uv"),
              "UV_PYTHON_INSTALL_DIR": str(ROOT / ".cache/python"), "UV_HTTP_TIMEOUT": "600"}
    for name, project in (("isaac-newton", ROOT / "training/isaac_newton"),
                          ("isaac-assets", ROOT / "training/isaac_newton/asset_converter")):
        env = {**shared, "UV_PROJECT_ENVIRONMENT": str(ROOT / ".envs" / name)}
        subprocess.run(["uv", "sync", "--project", str(project), "--locked", "--python", "3.12"], env=env, check=True)
        result = subprocess.check_output(["uv", "pip", "freeze", "--python", str(ROOT / ".envs" / name / "bin/python")], env=env, text=True)
        out = ROOT / "artifacts/environments" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "installed.txt").write_text(result)
        (out / "setup.json").write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(),
            "isaaclab": spec, "status": "installed_not_worker_validated"}, indent=2) + "\n")
    print("Isaac/Newton installed. Run asset conversion and validation through server jobs.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
