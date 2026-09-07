"""Launch one independent task/framework run per allocated GPU; no shared learner."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from oh_my_duck.core.paths import project_root


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema_version') != 1 or not plan.get('runs'):
        raise ValueError('Expected a nonempty schema-1 campaign')
    if plan.get('smoke_iterations', 0) < 5 or plan.get('checkpoint_interval', 0) < 1:
        raise ValueError('Campaign requires at least five smoke iterations and periodic checkpoints')
    from oh_my_duck.rl.training.tasks import project_tasks
    tasks = project_tasks()
    seen = set()
    for row in plan['runs']:
        if not re.fullmatch(r'[a-z0-9_-]+', row['id']) or row['id'] in seen:
            raise ValueError('Run IDs must be unique, filesystem-safe names')
        seen.add(row['id'])
        task = tasks.get(row['task'])
        task.binding(row['backend'])
        if task.evaluation is None or task.policy_package is None:
            raise ValueError('Full campaigns require task evaluation and policy-package profiles')
        if row['num_envs'] < 1 or row['iterations'] < 1:
            raise ValueError('Resource and iteration counts must be positive')
        if row['backend'] not in ('mujoco', 'isaac-newton') or row['framework'] not in ('rsl-rl', 'sb3'):
            raise ValueError('Unsupported native training combination')
    return plan


def assigned_devices(visible, count):
    devices = [s.strip() for s in visible.split(',') if s.strip()]
    if len(devices) < count or len(devices) != len(set(devices)):
        raise ValueError('Campaign needs a distinct allocated GPU for every run')
    # Pinned mjlab select_gpus consumes ordinal CUDA_VISIBLE_DEVICES entries.
    if not all(s.isdecimal() for s in devices):
        raise ValueError('Pinned native GPU selector requires numeric allocation ordinals')
    return devices[:count]


def worker_environment(device, group):
    environment = {k: v for k, v in os.environ.items() if k not in (
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')}
    environment.update(CUDA_VISIBLE_DEVICES=device, WANDB_MODE='offline',
                       WANDB_RUN_GROUP=group, PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = project_root()
    plan = load_plan(args.config)
    if args.dry_run:
        print(json.dumps({'required_gpus': len(plan['runs']), **plan}, indent=2))
        return 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        # The scheduler container may expose allocated GPUs without setting CVD.
        count = subprocess.check_output([str(root/'.envs/mujoco/bin/python'), '-c',
            'import torch; print(torch.cuda.device_count())'], text=True).strip()
        visible = ','.join(str(i) for i in range(int(count)))
    devices = assigned_devices(visible, len(plan['runs']))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report = {'source_commit': commit, 'status': 'running', 'devices': devices, 'plan': plan, 'runs': {}}
    children = []
    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def save():
        temporary = output/'campaign.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output/'campaign.json')
    def wait(row, child, log):
        try:
            return row['id'], child.wait()
        finally:
            log.close()
    save()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = []
        for row, device in zip(plan['runs'], devices):
            log = (output/(row['id']+'.log')).open('w')
            command = [sys.executable, '-m', 'oh_my_duck.rl.experiments.worker',
                '--config', str(args.config.resolve()), '--run-id', row['id'],
                '--output', str(output/row['id'])]
            child = subprocess.Popen(command, cwd=root, env=worker_environment(device, output.name),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            report['runs'][row['id']] = {'device': device, 'pid': child.pid, 'status': 'running'}
            futures.append(pool.submit(wait, row, child, log))
        save()
        for future in as_completed(futures):
            identity, code = future.result()
            report['runs'][identity].update(exit_code=code, status='completed' if code in (0, 2) else 'failed')
            save()
    codes = [row['exit_code'] for row in report['runs'].values()]
    report['status'] = 'execution_failed' if any(c not in (0, 2) for c in codes) else (
        'behavior_failed' if 2 in codes else 'passed')
    save()
    return 1 if report['status'] == 'execution_failed' else (2 if report['status'] == 'behavior_failed' else 0)


if __name__ == '__main__':
    raise SystemExit(main())
