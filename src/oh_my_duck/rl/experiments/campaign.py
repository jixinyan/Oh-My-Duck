"""Launch one independent task/framework run per allocated GPU; no shared learner."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
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
        if 'learning_rate' in row:
            rate = row['learning_rate']
            if row['framework'] != 'sb3':
                raise ValueError('Campaign learning_rate applies only to native SB3')
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ValueError('SB3 learning_rate must be a finite positive number')
    if search := plan.get('environment_search'):
        candidates = search.get('candidates', [])
        if not candidates or any(type(n) is not int or n < 64 for n in candidates) or candidates != sorted(set(candidates)):
            raise ValueError('Environment search needs increasing unique integer candidates >= 64')
        if not 0 < search.get('memory_fraction', 0) < 1 or not 0 < search.get('stop_below_best_fraction', 0) <= 1:
            raise ValueError('Environment search fractions must leave VRAM headroom')
        if not 1 <= search.get('warmup_updates', 0) < search.get('updates', 0):
            raise ValueError('Environment search requires warmup and measured updates')
        if any(row.get('resume') for row in plan['runs']):
            raise ValueError('Environment search requires fresh runs; resume cannot change vector size')
    return plan


def assigned_devices(visible, count, runs_per_gpu=1):
    devices = [s.strip() for s in visible.split(',') if s.strip()]
    if runs_per_gpu < 1:
        raise ValueError('runs-per-gpu must be positive')
    if len(devices) * runs_per_gpu < count or len(devices) != len(set(devices)):
        raise ValueError('Campaign needs a distinct allocated GPU for every run')
    # Pinned mjlab select_gpus consumes ordinal CUDA_VISIBLE_DEVICES entries.
    if not all(s.isdecimal() for s in devices):
        raise ValueError('Pinned native GPU selector requires numeric allocation ordinals')
    return [device for device in devices for _ in range(runs_per_gpu)][:count]


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
    parser.add_argument('--runs-per-gpu', type=int, default=1,
                        help='Explicit independent learner sharing; does not share gradients')
    parser.add_argument('--queue', action='store_true', help='Run sequentially on each allocated GPU when runs outnumber GPUs')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = project_root()
    plan = load_plan(args.config)
    if args.runs_per_gpu < 1:
        parser.error('--runs-per-gpu must be positive')
    if args.dry_run:
        print(json.dumps({'required_gpus': (len(plan['runs']) + args.runs_per_gpu - 1) // args.runs_per_gpu,
                          'runs_per_gpu': args.runs_per_gpu, **plan}, indent=2))
        return 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        # The scheduler container may expose allocated GPUs without setting CVD.
        count = subprocess.check_output([str(root/'.envs/mujoco/bin/python'), '-c',
            'import torch; print(torch.cuda.device_count())'], text=True).strip()
        visible = ','.join(str(i) for i in range(int(count)))
    if args.queue:
        if args.runs_per_gpu != 1:
            parser.error('Queued calibration/full runs require one worker per GPU')
        available = assigned_devices(visible, len(visible.split(',')))
        devices = [available[i % len(available)] for i in range(len(plan['runs']))]
    else:
        devices = assigned_devices(visible, len(plan['runs']), args.runs_per_gpu)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report = {'source_commit': commit, 'status': 'running', 'devices': devices, 'plan': plan,
              'runs_per_gpu': args.runs_per_gpu,
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(), 'runs': {}}
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
    lock = threading.Lock()
    lanes = {device: threading.Lock() for device in devices}
    stopping = threading.Event()
    original_stop = stop
    def stop(signum, frame):
        stopping.set()
        original_stop(signum, frame)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    for row, device in zip(plan['runs'], devices):
        report['runs'][row['id']] = {'device': device, 'status': 'queued'}
    save()
    def execute(row, device):
        # Distinct locks retain explicit simultaneous sharing when --queue is absent.
        lane = lanes[device] if args.queue else threading.Lock()
        with lane:
            if stopping.is_set():
                return row['id'], 143
            with (output/(row['id']+'.log')).open('x') as log:
                command = [sys.executable, '-m', 'oh_my_duck.rl.experiments.worker',
                    '--config', str(args.config.resolve()), '--run-id', row['id'],
                    '--output', str(output/row['id'])]
                if args.prepare_only:
                    command.append('--prepare-only')
                child = subprocess.Popen(command, cwd=root, env=worker_environment(device, output.name),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with lock:
                    children.append(child)
                    report['runs'][row['id']].update(pid=child.pid, status='running')
                    save()
                return row['id'], child.wait()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(execute, row, device) for row, device in zip(plan['runs'], devices)]
        for future in as_completed(futures):
            identity, code = future.result()
            with lock:
                report['runs'][identity].update(exit_code=code, status='completed' if code in (0, 2) else 'failed')
                save()
    codes = [row['exit_code'] for row in report['runs'].values()]
    report['status'] = 'execution_failed' if any(c not in (0, 2) for c in codes) else (
        'behavior_failed' if 2 in codes else 'passed')
    if args.prepare_only and report['status'] == 'passed':
        report['status'] = 'prepared'
    save()
    return 1 if report['status'] == 'execution_failed' else (2 if report['status'] == 'behavior_failed' else 0)


if __name__ == '__main__':
    raise SystemExit(main())
