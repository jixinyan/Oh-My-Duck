#!/usr/bin/env python3
"""Dispatch into a registered backend process; never import simulator dependencies here."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from oh_my_duck.core.paths import project_root
ROOT = project_root()
from oh_my_duck.rl.training.registry import BackendUnavailable, default_registry
from oh_my_duck.rl.training.frameworks import default_framework_registry


def main():
    from oh_my_duck.infrastructure.headless import configure_egl
    configure_egl()
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("action", choices=["assets", "probe", "train", "export", "eval", "compare", "rehearsal", "package"])
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--rl-framework", default=None)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        if args.action in {'compare','rehearsal','package'}:
            from oh_my_duck.rl.training.base import BackendCommand
            entry={'compare':'oh_my_duck.rl.evaluation.sim2sim', 'rehearsal':'oh_my_duck.rl.evaluation.rehearsal.task',
                   'package':'oh_my_duck.rl.artifacts.package_policy'}[args.action]
            interpreter=sys.executable if args.action=='compare' else str(ROOT/'.envs/mujoco/bin/python')
            command=BackendCommand((interpreter,'-m',entry,*extra),ROOT,
                {'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'egl','MPLBACKEND':'Agg','WANDB_MODE':'offline'})
        elif args.action in {"train", "export"}:
            framework = args.rl_framework or "rsl-rl"
            command = default_framework_registry().command(framework, args.backend, ROOT, args.action, extra)
        else:
            if args.rl_framework is not None:
                raise BackendUnavailable("--rl-framework applies to train/export; assets, probe and ONNX eval are framework-independent")
            command = default_registry().get(args.backend, ROOT).command(args.action, extra)
    except BackendUnavailable as error:
        parser.error(str(error))
    if args.action in {"train", "export"}:
        print("RL framework:", framework, flush=True)
    env = {**os.environ, **command.environment, "PYTHONUNBUFFERED": "1"}
    print("Backend:", args.backend, "Action:", args.action, flush=True)
    return subprocess.call(command.argv, cwd=command.cwd, env=env)


if __name__ == "__main__":
    sys.exit(main())
