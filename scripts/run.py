#!/usr/bin/env python3
"""Dispatch into a registered backend process; never import simulator dependencies here."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oh_my_duck.training.registry import BackendUnavailable, default_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("action", choices=["probe", "train", "export", "eval"])
    parser.add_argument("--backend", default="mujoco")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        backend = default_registry().get(args.backend, ROOT)
        command = backend.command(args.action, extra)
    except BackendUnavailable as error:
        parser.error(str(error))
    env = {**os.environ, **command.environment}
    print("Backend:", args.backend, "Action:", args.action, flush=True)
    return subprocess.call(command.argv, cwd=command.cwd, env=env)


if __name__ == "__main__":
    sys.exit(main())
