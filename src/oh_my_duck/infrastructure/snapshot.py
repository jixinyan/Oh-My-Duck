"""Immutable committed source for scheduled jobs that outlive development edits."""
from pathlib import Path
import subprocess

GENERATED = ('.envs', '.cache', 'artifacts', 'outputs', 'logs', 'wandb')


def source_snapshot(root: Path) -> tuple[Path,str]:
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    if dirty:
        raise RuntimeError('Commit the source before submitting a reproducible job')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    # Keep the snapshot outside .cache: its .cache link points at the shared cache.
    directory=root/'.job-sources'/revision
    if not directory.exists():
        directory.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','worktree','add','--detach',str(directory),revision],cwd=root,check=True,capture_output=True,text=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
    if actual!=revision:raise RuntimeError('Job snapshot revision mismatch')
    for name in GENERATED:
        target=root/name
        target.mkdir(exist_ok=True)
        link=directory/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        if link.resolve()!=target.resolve():raise RuntimeError(f'Unexpected snapshot output link: {link}')
    if subprocess.check_output(['git','status','--porcelain'],cwd=directory,text=True).strip():
        raise RuntimeError('Job source snapshot was modified')
    return directory,revision
