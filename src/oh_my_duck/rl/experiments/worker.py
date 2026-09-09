"""Smoke, capacity gate, full native training and task-specific evaluation on one GPU."""
import argparse
import hashlib
import os
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
from oh_my_duck.core.paths import project_root
from .campaign import load_plan, SB3_OPTIONS
from .gpu_pool import gpu_lease


def native_run_tag(output, identity, stage):
    campaign_key = hashlib.sha256(str(output.parent.resolve()).encode()).hexdigest()[:12]
    return f'{output.parent.name}-{campaign_key}_{identity}_{stage}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Run gates and optional throughput selection without full training')
    parser.add_argument('--prepared', type=Path, help='Reuse verified completed preparation from an earlier campaign')
    args = parser.parse_args()
    plan = load_plan(args.config)
    spec = next(row for row in plan['runs'] if row['id'] == args.run_id)
    root, output = project_root(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'spec': spec, 'status': 'running', 'stages': {}}
    prefix = [sys.executable, str(root/'omd.py')]
    def save():
        temporary = output/'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output/'result.json')
    def stage(name, command, allowed=(0,)):
        report['stages'][name] = {'status': 'running', 'command': command}
        save()
        started = time.monotonic()
        cpu_only = name in ('smoke-rehearsal', 'rehearsal', 'package')
        with nullcontext() if cpu_only else gpu_lease() as device:
            report['stages'][name]['gpu'] = None if cpu_only else device
            save()
            with (output/(name+'.log')).open('w') as stream:
                if cpu_only:
                    code = subprocess.call(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                else:
                    code = subprocess.call(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        report['stages'][name].update(status='completed' if code in allowed else 'failed',
                                      exit_code=code, wall_time_s=time.monotonic()-started)
        save()
        if code not in allowed:
            raise RuntimeError(f'{name} exited {code}; see {output/(name+".log")}')
        return code
    def train(name, count, iterations, resume=None, resume_checkpoint=None):
        command = prefix+['train', '--backend', spec['backend'], '--rl-framework', spec['framework'], '--', spec['task']]
        if spec['framework'] == 'sb3':
            run = output/name
            command += ['--num-envs', str(count), '--iterations', str(iterations), '--seed', str(spec['seed']),
                        '--checkpoint-interval', str(2 if name == 'smoke' else plan['checkpoint_interval']), '--output', str(run)]
            if 'learning_rate' in spec:
                command += ['--learning-rate', str(spec['learning_rate'])]
            for key in SB3_OPTIONS:
                if key in spec:
                    command += ['--' + key.replace('_', '-'), spec[key]]
            if resume is not None:
                bundles = sorted((resume/'checkpoints').glob('step_*'))
                command += ['--resume', str(bundles[-1] if bundles else resume)]
        else:
            tag = native_run_tag(output, spec['id'], name)
            command += ['--env.scene.num-envs', str(count), '--agent.max-iterations', str(iterations),
                        '--agent.seed', str(spec['seed']), '--agent.run-name', tag,
                        '--agent.save-interval', str(plan['checkpoint_interval']), '--agent.upload-model', 'False']
            if resume is not None:
                command += ['--agent.resume', 'True', '--agent.load-run', resume.name,
                            '--agent.load-checkpoint', resume_checkpoint or latest_checkpoint(resume).name]
        stage(name, command)
        if spec['framework'] == 'rsl-rl':
            matches = list((root/'logs/rsl_rl'/spec['experiment']).glob('*_'+tag))
            if len(matches) != 1:
                raise RuntimeError(f'Expected one native run directory, found {matches}')
            run = matches[0].resolve()
        evidence = json.loads((run/'run.json').read_text())
        report['stages'][name]['run'] = str(run)
        report['stages'][name]['training_wall_time_s'] = evidence['wall_time_s']
        report['stages'][name]['transitions_per_second_including_startup'] = count*iterations*24/evidence['wall_time_s']
        save()
        return run
    def latest_checkpoint(run):
        return max(run.glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
    def export(name, run):
        dest = output/name
        if spec['framework'] == 'rsl-rl':
            checkpoint = latest_checkpoint(run)
            command = [str(root/'.envs/mujoco/bin/python'), '-m', 'oh_my_duck.rl.evaluation.verify_rsl',
                spec['task'], '--run', str(run), '--checkpoint', checkpoint.name, '--output', str(dest)]
        else:
            checkpoint = run/'model.zip'
            command = prefix+['export', '--backend', spec['backend'], '--rl-framework', 'sb3', '--',
                '--run', str(run), '--output', str(dest)]
        stage(name, command)
        return dest/'policy.onnx', checkpoint
    def rehearsal(name, policy):
        return stage(name, prefix+['rehearsal', '--', '--task', spec['task'], '--policy', str(policy),
            '--output', str(output/name), '--video', '--mujoco-renderer', 'osmesa'], allowed=(0, 2))
    save()
    try:
        if args.prepared:
            from .preparation import validate_preparation
            report['preparation'] = validate_preparation(args.prepared, spec, root)
            save()
            run = train('full', spec['num_envs'], spec['iterations'])
        elif spec.get('preparation_source'):
            from .preparation import reuse_smoke
            report['preparation'] = reuse_smoke(spec['preparation_source'], spec, root)
            save()
            previous_capacity = report['preparation']['capacity_run']
            capacity = Path(previous_capacity) if previous_capacity else train('capacity', spec['num_envs'], 5)
            export('capacity-export', capacity)
            run = train('full', spec['num_envs'], spec['iterations'])
        elif spec.get('resume'):
            from .recovery import validate_checkpoint
            recovery = validate_checkpoint(spec, root)
            report['recovery'] = recovery
            save()
            run = train('full', spec['num_envs'], recovery['remaining_iterations'],
                        Path(recovery['run']), recovery['checkpoint'])
        else:
            smoke = train('smoke', 64, plan['smoke_iterations'])
            smoke_policy, _ = export('smoke-export', smoke)
            rehearsal('smoke-rehearsal', smoke_policy)
            train('resume-check', 64, 1, smoke)
            if search := plan.get('environment_search'):
                from .scaling import GpuSampler, choose, measure
                rows = report['environment_search'] = []
                for count in search['candidates']:
                    # A conservative linear extrapolation prevents knowingly filling VRAM.
                    if rows and rows[-1]['gpu']['peak_mib'] * count / rows[-1]['num_envs'] > rows[-1]['gpu']['total_mib'] * search['memory_fraction']:
                        report['environment_search_stop'] = 'next_count_exceeds_projected_vram_budget'
                        break
                    name = f'scaling-{count}'
                    with gpu_lease(), GpuSampler() as sampler:
                        candidate = train(name, count, search['updates'])
                    row = {'num_envs': count, 'run': str(candidate), 'gpu': sampler.data,
                           'measurement': measure(candidate, output/(name+'.log'), spec['framework'], count,
                                                  search['warmup_updates'], search['updates'])}
                    rows.append(row)
                    save()
                    if sampler.data['foreign_pids'] or sampler.data['errors']:
                        raise RuntimeError('Contaminated throughput benchmark; preserved, no automatic retry')
                    best = choose(rows, search['memory_fraction'])
                    if row['measurement']['samples_per_second'] < best['measurement']['samples_per_second'] * search['stop_below_best_fraction']:
                        report['environment_search_stop'] = 'throughput_regressed'
                        break
                best = choose(rows, search['memory_fraction'])
                spec['num_envs'] = best['num_envs']
                report['environment_selection'] = best
                save()
            capacity = train('capacity', spec['num_envs'], 5)
            export('capacity-export', capacity)
            if args.prepare_only:
                report['status'] = 'prepared'
                save()
                return 0
            run = train('full', spec['num_envs'], spec['iterations'])
        policy, checkpoint = export('export', run)
        stage('package', prefix+['package', '--', '--onnx', str(policy), '--checkpoint', str(checkpoint),
                                '--output', str(output/'package')])
        comparison = stage('sim2sim', prefix+['compare', '--', '--task', spec['task'], '--policy', str(policy),
            '--output', str(output/'sim2sim'), '--video', '--mujoco-renderer', 'osmesa'], allowed=(0, 2))
        deployment = rehearsal('rehearsal', policy)
        report['status'] = 'passed' if comparison == deployment == 0 else 'behavior_failed'
    except Exception as error:
        report.update(status='execution_failed', error=repr(error))
        raise
    finally:
        save()
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
