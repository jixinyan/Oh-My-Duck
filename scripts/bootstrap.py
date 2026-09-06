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
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def run(argv, **kwargs):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-env", action="store_true")
    args = parser.parse_args()
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
        env.update(UV_PROJECT_ENVIRONMENT=str(ROOT / ".envs/mujoco"),
                   UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                   UV_PYTHON_INSTALL_DIR=str(ROOT / ".cache/python"), UV_HTTP_TIMEOUT="600")
        run(["uv", "sync", "--project", upstream / "microduck_rl", "--locked", "--python", "3.12"], env=env)
    print("Bootstrap complete. GPU validation must run through the scheduler.")


if __name__ == "__main__":
    main()
