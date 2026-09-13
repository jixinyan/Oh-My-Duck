"""Render periodic checkpoint snapshots; training itself stays headless."""
import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from oh_my_duck.core.paths import project_root
from .campaign import worker_environment


def checkpoint_source(spec, campaign, root, minimum_updates=0):
    """Prefer a completed periodic save; recoveries can preview their source save."""
    result = campaign / spec['id'] / 'result.json'
    if result.exists():
        stage = json.loads(result.read_text()).get('stages', {}).get('full', {})
        command = stage.get('command', [])
        if spec['framework'] == 'sb3' and '--output' in command:
            directory = Path(command[command.index('--output') + 1])
            bundles = sorted((directory / 'checkpoints').glob('step_*'), reverse=True)
            if (directory / 'model.zip').exists() and (directory / 'run.json').exists():
                bundles.insert(0, directory)
            for bundle in bundles:
                if all((bundle / name).exists() for name in ('model.zip', 'vecnormalize.pkl', 'run.json')):
                    if minimum_updates:
                        metadata = json.loads((bundle/'run.json').read_text())
                        if metadata['timesteps_after'] < minimum_updates * metadata['num_envs'] * 24:
                            continue
                    return bundle, bundle / 'model.zip'
        elif '--agent.run-name' in command:
            tag = command[command.index('--agent.run-name') + 1]
            directories = list((root / 'logs/rsl_rl' / spec['experiment']).glob('*_' + tag))
            if len(directories) == 1:
                saves = sorted(directories[0].glob('model_*.pt'),
                               key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
                for save in saves:
                    if int(save.stem.split('_')[-1]) >= minimum_updates and time.time() - save.stat().st_mtime >= 15:
                        return directories[0], save
    recovery = spec.get('resume')
    if recovery:
        directory = Path(recovery['run'])
        return directory, directory / recovery['checkpoint']
    return None


def gallery(output, report):
    cards = []
    for name, entry in report['runs'].items():
        videos = []
        for video in sorted((output / entry['directory']).glob('evaluation/*.mp4')):
            relative = video.relative_to(output).as_posix()
            videos.append(f'<figure><video controls preload="metadata" src="{html.escape(relative)}"></video>'
                          f'<figcaption>{html.escape(video.stem)}</figcaption></figure>')
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(entry["checkpoint"])} · '
                     f'{html.escape(entry["status"])} · {html.escape(entry["captured_at_utc"])}</p>'
                     + ''.join(videos) + '</section>')
    text = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Microduck training previews</title>
<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#101b2b;color:#edf4ff}
section{background:#1b2b40;padding:20px;margin:20px 0;border-radius:12px}
figure{display:inline-block;width:min(100%,540px);margin:10px}video{width:100%}p{color:#b9cce2}</style>
<h1>Microduck training previews</h1><p>Saved checkpoint snapshots, not a live camera.
Refresh to see newly completed previews. Behavior scores are separate from training reward.</p>'''
    temporary = output / 'index.html.tmp'
    temporary.write_text(text + ''.join(cards) + '</html>')
    temporary.replace(output / 'index.html')


@contextmanager
def training_preview(root, output, spec, device, checkpoint_interval):
    """Follow one learner on its allocated GPU; finish with that training stage.

    The follower inherits the worker process group so scheduler cancellation
    reaches rendering/export subprocesses too. A stop marker avoids waiting for
    the whole campaign (which would deadlock workers waiting for previews).
    """
    stop_file = output / 'preview-training-finished'
    destination = output / 'previews'
    command = [sys.executable, str(root / 'omd.py'), 'preview',
               '--campaign', str(output.parent), '--output', str(destination),
               '--run-id', spec['id'], '--gpu', str(device), '--watch',
               '--minimum-updates', str(checkpoint_interval), '--stop-file', str(stop_file)]
    record = {'status': 'running', 'output': str(destination), 'command': command,
              'checkpoint_interval_updates': checkpoint_interval, 'gpu': device}
    with (output / 'preview.log').open('x') as stream:
        follower = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        record['pid'] = follower.pid
        try:
            yield record
        finally:
            stop_file.write_text('Full training stage ended; final evaluation is a separate pipeline stage.\n')
            record['exit_code'] = follower.wait()
            record['status'] = 'completed' if record['exit_code'] == 0 else 'preview_error'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--run-id', action='append')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--stop-file', type=Path, help='Stop the follower when its own training stage finishes')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--minimum-updates', type=int, default=0, help='Skip initial untrained checkpoints')
    args = parser.parse_args()
    if not args.gpu.isdecimal() or args.interval <= 0 or args.minimum_updates < 0:
        parser.error('Use a numeric GPU ordinal and a positive polling interval')
    root, output = project_root(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'checkpoint_preview', 'gpu': args.gpu, 'runs': {}, 'attempts': []}
    environment = worker_environment(args.gpu, output.name)
    seen = set()
    def save():
        temporary = output / 'previews.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'previews.json')
        gallery(output, report)
    def execute(command, log):
        with log.open('x') as stream:
            code = subprocess.call(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        return code
    save()
    while True:
        if args.stop_file and args.stop_file.exists():
            break
        manifest = json.loads((args.campaign / 'campaign.json').read_text())
        specs = manifest['plan']['runs']
        if args.run_id:
            known = {spec['id'] for spec in specs}
            if set(args.run_id) - known:
                parser.error('Unknown run ID')
            specs = [spec for spec in specs if spec['id'] in args.run_id]
        for spec in specs:
            source = checkpoint_source(spec, args.campaign, root, args.minimum_updates)
            if source is None:
                continue
            run, checkpoint = source
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            identity = (spec['id'], str(checkpoint), digest)
            if identity in seen:
                continue
            seen.add(identity)
            directory = output / spec['id'] / (checkpoint.stem + '-' + digest[:12])
            directory.mkdir(parents=True, exist_ok=False)
            from datetime import datetime, timezone
            entry = {'checkpoint': str(checkpoint), 'sha256': digest,
                     'directory': str(directory.relative_to(output)), 'status': 'rendering',
                     'captured_at_utc': datetime.now(timezone.utc).isoformat()}
            report['runs'][spec['id']] = entry
            report['attempts'].append(entry)
            save()
            if spec['framework'] == 'rsl-rl':
                command = [str(root / '.envs/mujoco/bin/python'), '-m', 'oh_my_duck.rl.evaluation.verify_rsl',
                           spec['task'], '--run', str(run), '--checkpoint', checkpoint.name,
                           '--output', str(directory / 'export')]
            else:
                command = [sys.executable, str(root / 'omd.py'), 'export', '--backend', spec['backend'],
                           '--rl-framework', 'sb3', '--', '--run', str(run), '--output', str(directory / 'export')]
            entry['export_exit_code'] = execute(command, directory / 'export.log')
            if entry['export_exit_code'] == 0:
                command = [sys.executable, str(root / 'omd.py'), 'eval', '--backend', spec['backend'], '--',
                           '--task', spec['task'], '--policy', str(directory / 'export/policy.onnx'),
                           '--output', str(directory / 'evaluation'), '--video', '--mujoco-renderer', 'osmesa']
                entry['evaluation_exit_code'] = execute(command, directory / 'evaluation.log')
                entry['status'] = {0: 'passed', 2: 'behavior_failed'}.get(entry['evaluation_exit_code'], 'evaluation_error')
            else:
                entry['status'] = 'export_error'
            save()
        if not args.watch or manifest['status'] != 'running' or (args.stop_file and args.stop_file.exists()):
            break
        time.sleep(args.interval)
    return int(any(row['status'].endswith('_error') for row in report['runs'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
