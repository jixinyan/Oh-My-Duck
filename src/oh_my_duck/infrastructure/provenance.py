"""Capture local source and dependency identity without reading credentials."""
import hashlib
import importlib.metadata
import json
import subprocess
from oh_my_duck.core.paths import project_root


def source_provenance():
    root = project_root()
    digest = hashlib.sha256()
    paths = sorted([*root.joinpath('src').rglob('*.py'), *root.joinpath('configs').glob('*.json')])
    for path in paths:
        digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
    packages = {}
    for name in ('torch','mjlab','rsl-rl-lib','stable-baselines3','newton','warp-lang','mujoco','mujoco-warp','wandb','tyro'):
        try:packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'working_tree':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).splitlines(),
        'source_config_sha256':digest.hexdigest(), 'packages':packages,
        'upstream':json.loads((root/'configs/upstream.json').read_text())['repositories']}


def validate_resume(run, *, task, backend, framework):
    for key, expected in [('task',task),('backend',backend),('framework',framework)]:
        if run.get(key) != expected:
            raise ValueError(f'Resume {key} mismatch: expected {expected!r}, got {run.get(key)!r}')
