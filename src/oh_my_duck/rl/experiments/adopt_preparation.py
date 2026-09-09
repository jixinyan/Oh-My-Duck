"""Finish an existing preparation campaign and start each ready learner independently."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from oh_my_duck.core.paths import project_root
from .campaign import worker_environment
from .preparation import validate_preparation


def free_devices(devices, preparation_runs, training_runs):
    # Old queued workers still own the allocator: wait until they are assigned,
    # avoiding a race with its next GPU assignment. Running CPU stages in this
    # legacy campaign retain their old reservation; new campaigns use stage leases.
    if any(r['status'] == 'queued' for r in preparation_runs.values()):
        return []
    occupied = {r['device'] for r in preparation_runs.values() if r['status'] == 'running'}
    occupied.update(r['device'] for r in training_runs.values() if r['status'] == 'running')
    return [d for d in devices if d not in occupied]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True, help='Existing throughput supervisor output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-gpu', default='0')
    args = parser.parse_args()
    root, prior = project_root(), args.preparation.resolve()
    old = json.loads((prior/'result.json').read_text())
    identity = json.loads(prior.with_suffix('.launch.json').read_text())
    stat = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    if stat[19] != identity['start_ticks'] or stat[0] not in ('T', 't'):
        raise RuntimeError('Original barrier supervisor must be identity-checked and paused before handover')
    campaign = prior/'prepare'
    manifest = json.loads((campaign/'campaign.json').read_text())
    devices = manifest['devices']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    full = output/'full'
    full.mkdir()
    plan = dict(manifest['plan'])
    plan.pop('environment_search', None)
    report = {'status': 'running', 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'preparation_source': str(prior), 'plan': plan, 'devices': devices,
              'runs': {s['id']: {'status': 'awaiting_preparation'} for s in plan['runs']}}
    children = {}
    old_released = False
    def save():
        temp = full/'campaign.json.tmp'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(full/'campaign.json')
    def release_old():
        nonlocal old_released
        if old_released:
            return
        path = Path(f'/proc/{identity["pid"]}/stat')
        if path.exists() and path.read_text().rsplit(')', 1)[1].split()[19] == identity['start_ticks']:
            os.kill(identity['pid'], signal.SIGTERM)
            os.kill(identity['pid'], signal.SIGCONT)
        old_released = True
        report['previous_supervisor'] = 'superseded_preview_resumed'
    def stop(signum, frame):
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signum)
        release_old()
        report['status'] = 'interrupted'
        save()
        raise SystemExit(128+signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {a.strip(): b.strip() for a,b in (line.split(',') for line in gpu_rows.splitlines())}
    save()
    with (output/'previews.log').open('x') as log:
        preview = subprocess.Popen([sys.executable, '-m', 'oh_my_duck.rl.experiments.preview',
            '--campaign', str(full), '--output', str(output/'previews'), '--gpu', args.preview_gpu, '--watch'],
            cwd=root, env=worker_environment(args.preview_gpu, output.name), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    children['preview'] = preview
    try:
        while True:
            manifest = json.loads((campaign/'campaign.json').read_text())
            for name, child in list(children.items()):
                if name != 'preview' and report['runs'][name]['status'] == 'running' and child.poll() is not None:
                    report['runs'][name].update(status='completed' if child.returncode in (0,2) else 'failed', exit_code=child.returncode)
            if manifest['status'] != 'running':
                release_old()
            available = free_devices(devices, manifest['runs'], report['runs'])
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
            busy = set(apps.splitlines())
            available = [d for d in available if uuids[d] not in busy]
            for i, requested in enumerate(plan['runs']):
                name = requested['id']
                row = report['runs'][name]
                if row['status'] != 'awaiting_preparation':
                    continue
                path = campaign/name/'result.json'
                if not path.exists():
                    continue
                prepared = json.loads(path.read_text())
                if prepared['status'] == 'execution_failed':
                    row.update(status='preparation_failed', error=prepared.get('error'))
                    continue
                if prepared['status'] != 'prepared' or not available:
                    continue
                spec = prepared['spec']
                evidence = validate_preparation(path, spec, root)
                plan['runs'][i] = spec
                config = output/(name+'.json')
                config.write_text(json.dumps({**plan, 'runs': [spec]}, indent=2)+'\n')
                device = available.pop(0)
                command = [sys.executable, '-m', 'oh_my_duck.rl.experiments.worker', '--config', str(config),
                    '--run-id', name, '--output', str(full/name), '--prepared', str(path)]
                with (full/(name+'.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=root, env=worker_environment(device, output.name),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children[name] = child
                row.update(status='running', pid=child.pid, device=device, num_envs=spec['num_envs'], preparation=evidence)
                save()
            save()
            if all(r['status'] in ('completed','failed','preparation_failed') for r in report['runs'].values()):
                break
            time.sleep(5)
        report['status'] = 'execution_failed' if any(r['status'] != 'completed' for r in report['runs'].values()) else (
            'behavior_failed' if any(r['exit_code'] == 2 for r in report['runs'].values()) else 'passed')
        save()
        report['preview_exit_code'] = preview.wait()
    except Exception as error:
        report.update(status='execution_failed', error=repr(error))
        # Keep already-running learners and all artifacts; do not retry or kill them.
        raise
    finally:
        save()
    return int(report['status'] == 'execution_failed')


if __name__ == '__main__':
    raise SystemExit(main())
