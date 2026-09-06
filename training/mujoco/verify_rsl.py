"""Audit a native official RSL run and export with the official runner."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.export import run_export, ExportConfig
from mjlab_microduck.publish.manifest import check_onnx, smoke_run_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='model_4.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_env_cfg(args.task)
    accumulator = EventAccumulator(str(args.run), size_guidance={'scalars': 0}).Reload()
    ranges = {}
    for tag in accumulator.Tags()['scalars']:
        values = np.asarray([x.value for x in accumulator.Scalars(tag)])
        if not len(values) or not np.isfinite(values).all():
            raise AssertionError(f'Invalid logged scalar: {tag}')
        ranges[tag] = {'count': len(values), 'min': float(values.min()), 'max': float(values.max())}
    if not ranges:
        raise AssertionError('No scalar evidence')
    penalties = []
    for key, term in cfg.rewards.items():
        if term is None:
            continue
        tag = 'Episode_Reward/' + key
        if tag not in ranges:
            raise AssertionError(f'Missing reward evidence: {key}')
        if term.weight < 0 or getattr(term.func, '__name__', '').endswith(('_penalty', '_l1')):
            if ranges[tag]['max'] > 1e-7:
                raise AssertionError(f'Positive weighted penalty: {key}')
            penalties.append(key)
    audit = {'status': 'passed', 'task': args.task, 'run': str(args.run.resolve()),
             'scalars': ranges, 'penalties_checked': penalties,
             'behavior_validation': 'pending'}
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    result = run_export(args.task, ExportConfig(checkpoint_file=str(args.run / args.checkpoint),
        onnx_file=str(args.output / 'policy.onnx'), num_envs=1, device='cuda:0'))
    check_onnx(result.onnx_path)
    smoke_run_onnx(result.onnx_path)
    (args.output / 'result.json').write_text(json.dumps({'status': 'passed', 'task': args.task,
        'export': str(result.onnx_path), 'scalar_tags': len(ranges), 'penalties': len(penalties),
        'behavior_validation': 'pending'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
