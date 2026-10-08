"""Prepare the isolated, pinned Isaac Lab / Newton environment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def sync_environment(name, project, environment, rl_framework):
    extras = ["--extra", "native-harness"] if name == "isaac-newton" else []
    if name == "isaac-newton" and rl_framework == "sb3":
        extras.extend(["--extra", "sb3"])
    reinstall = ["--reinstall-package", "usd-core"] if name == "isaac-newton" else []
    subprocess.run(["uv", "sync", "--project", str(project), "--locked", "--python", "3.12",
                    *extras, *reinstall], env=environment, check=True)
    if name != "isaac-newton":
        return None
    python = Path(environment["UV_PROJECT_ENVIRONMENT"]) / "bin/python"
    return json.loads(subprocess.check_output([
        str(python), "-c",
        "import json; from oh_my_duck.infrastructure.usd_runtime import verify_usd_runtime; print(json.dumps(verify_usd_runtime()))",
    ], cwd=ROOT, env=environment, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-env", action="store_true")
    parser.add_argument("--rl-framework", choices=["rsl-rl", "sb3"], default="rsl-rl")
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
    subprocess.run([sys.executable, "-m", "oh_my_duck.infrastructure.bootstrap", "--skip-env"], check=True)
    if args.skip_env:
        return 0
    temporary = ROOT / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    shared = {**os.environ, "UV_CACHE_DIR": str(ROOT / ".cache/uv"),
              "UV_PYTHON_INSTALL_DIR": str(ROOT / ".cache/python"), "UV_HTTP_TIMEOUT": "600",
              "TMPDIR": str(temporary)}
    for name, project in (("isaac-newton", ROOT / "environments/isaac-newton"),
                          ("isaac-assets", ROOT / "environments/isaac-assets")):
        env = {**shared, "UV_PROJECT_ENVIRONMENT": str(ROOT / ".envs" / name)}
        openusd = sync_environment(name, project, env, args.rl_framework)
        result = subprocess.check_output(["uv", "pip", "freeze", "--python", str(ROOT / ".envs" / name / "bin/python")], env=env, text=True)
        out = ROOT / "artifacts/environments" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "installed.txt").write_text(result)
        (out / "setup.json").write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(),
            "isaaclab": spec, "rl_framework": args.rl_framework if name == "isaac-newton" else None,
            "openusd": openusd, "status": "installed_not_worker_validated"}, indent=2) + "\n")
    print("Isaac/Newton installed. Run asset conversion and validation through server jobs.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
