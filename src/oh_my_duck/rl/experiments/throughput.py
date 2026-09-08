"""Prepare every combination, then train with its measured environment count."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from oh_my_duck.core.paths import project_root
from .campaign import load_plan, worker_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--preview-gpu', required=True)
    parser.add_argument('--pause-preview-launch', type=Path, help='Temporarily pause an identified background preview during isolated calibration')
    args = parser.parse_args()
    root = project_root()
    plan = load_plan(args.config)
    if 'environment_search' not in plan:
        parser.error('An environment_search plan is required')
    if not all(x.isdecimal() for x in args.gpus.split(',')) or not args.preview_gpu.isdecimal():
        parser.error('Use explicit numeric GPU ordinals')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'gpus': args.gpus, 'preview_gpu': args.preview_gpu,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}
    environment = worker_environment(args.gpus, output.name)
    children = []
    paused = None
    def resume_preview():
        nonlocal paused
        if paused is not None:
            stat = Path(f'/proc/{paused["pid"]}/stat')
            if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[19] == paused['start_ticks']:
                os.killpg(paused['pid'], signal.SIGCONT)
            report['preview_pause']['status'] = 'resumed'
            paused = None

    def save():
        temporary = output / 'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'result.json')
    def stop(signum, frame):
        # Campaign owns its worker process groups and forwards termination.
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)
        resume_preview()
        report['status'] = 'interrupted'
        save()
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def launch(name, command):
        with (output / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        report[name] = {'pid': child.pid, 'command': command}
        save()
        return child
    save()
    try:
        if args.pause_preview_launch:
            identity = json.loads(args.pause_preview_launch.read_text())
            pid = identity['pid']
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if stat[19] != str(identity['start_ticks']) or os.getpgid(pid) != pid or '/outputs/previews/' not in command:
                raise RuntimeError('Preview process identity does not match; refusing to signal')
            paused = {'pid': pid, 'start_ticks': str(identity['start_ticks'])}
            os.killpg(pid, signal.SIGSTOP)
            report['preview_pause'] = {**paused, 'status': 'paused_for_isolated_calibration'}
            save()
        prepare = launch('prepare', [sys.executable, '-m', 'oh_my_duck.rl.experiments.campaign',
            '--config', str(args.config.resolve()), '--output', str(output/'prepare'), '--queue', '--prepare-only'])
        code = prepare.wait()
        if code:
            raise RuntimeError(f'Preparation failed ({code}); no automatic full training or retry')
        resume_preview()
        selected = dict(plan)
        selected.pop('environment_search')
        selected['runs'] = []
        for spec in plan['runs']:
            evidence = json.loads((output/'prepare'/spec['id']/'result.json').read_text())
            if evidence['status'] != 'prepared' or 'environment_selection' not in evidence:
                raise RuntimeError('Every combination must pass preparation before full launch')
            selected['runs'].append(evidence['spec'])
        selected['selection_evidence'] = str(output/'prepare')
        config = output/'selected.json'
        config.write_text(json.dumps(selected, indent=2)+'\n')
        report['status'] = 'training'
        full = launch('full', [sys.executable, '-m', 'oh_my_duck.rl.experiments.campaign',
            '--config', str(config), '--output', str(output/'full'), '--queue'])
        while not (output/'full/campaign.json').exists():
            if full.poll() is not None:
                raise RuntimeError('Full campaign exited before writing its manifest')
            time.sleep(1)
        previews = launch('previews', [sys.executable, '-m', 'oh_my_duck.rl.experiments.preview',
            '--campaign', str(output/'full'), '--output', str(output/'previews'),
            '--gpu', args.preview_gpu, '--watch'])
        report['full_exit_code'] = full.wait()
        report['preview_exit_code'] = previews.wait()
        report['status'] = json.loads((output/'full/campaign.json').read_text())['status']
    except Exception as error:
        report.update(status='execution_failed', error=repr(error))
        raise
    finally:
        resume_preview()
        save()
    return report['full_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
