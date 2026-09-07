#!/usr/bin/env python3
"""Fetch pinned upstreams/models and create an isolated official RL environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from oh_my_duck.core.paths import project_root
ROOT = project_root()


def run(argv, **kwargs):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-env", action="store_true")
    parser.add_argument("--backend", choices=["mujoco", "isaac-newton"], default="mujoco")
    parser.add_argument("--rl-framework", choices=["rsl-rl", "sb3"], default="rsl-rl")
    parser.add_argument("--software-renderer", action="store_true", help="Install pinned local OSMesa for headless video on Ubuntu 22.04 amd64")
    args = parser.parse_args()
    if args.software_renderer:
        from oh_my_duck.infrastructure.headless import install_osmesa
        install_osmesa()
    if args.backend == "isaac-newton":
        run([sys.executable, "-m", "oh_my_duck.infrastructure.bootstrap_isaac", "--rl-framework", args.rl_framework, *(["--skip-env"] if args.skip_env else [])])
        return
    lock = json.loads((ROOT / "configs/upstream.json").read_text())
    upstream = ROOT / ".cache/upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    for name, spec in lock["repositories"].items():
        dest = upstream / name
        if not dest.exists():
            run(["git", "clone", "--no-checkout", "--filter=blob:none", spec["url"], dest])
            run(["git", "checkout", "--detach", spec["commit"]], cwd=dest)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=dest, text=True).strip()
        if actual != spec["commit"] or dirty:
            raise SystemExit(f"{dest} does not match the clean pinned commit; inspect it before continuing")
    policy = lock["policy"]
    dest = ROOT / "artifacts/policies/official" / policy["revision"]
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in policy["files"]:
        target = dest / name
        if not target.exists():
            url = f"https://huggingface.co/{policy['repo']}/resolve/{policy['revision']}/{name}"
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / "provenance.json").write_text(json.dumps({**policy, "sha256": hashes}, indent=2) + "\n")
    if not args.skip_env:
        env = os.environ.copy()
        env.update(UV_PROJECT_ENVIRONMENT=str(ROOT / (".envs/mujoco-sb3" if args.rl_framework == "sb3" else ".envs/mujoco")),
                   UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                   UV_PYTHON_INSTALL_DIR=str(ROOT / ".cache/python"), UV_HTTP_TIMEOUT="600")
        project = ROOT / "environments/mujoco"
        extras = ["--extra", "sb3"] if args.rl_framework == "sb3" else []
        run(["uv", "sync", "--project", project, "--locked", "--python", "3.12", *extras], env=env)
    print("Bootstrap complete. Run single-GPU validation locally; schedule multi-GPU experiments.")


if __name__ == "__main__":
    main()
