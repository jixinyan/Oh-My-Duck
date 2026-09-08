"""Throughput selection excludes startup, incomplete timing and contaminated GPUs."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from oh_my_duck.rl.experiments.scaling import measure, choose


def test_native_rsl_warmup_is_excluded():
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / 'train.log'
        log.write_text('Iteration time: 90.0\n' * 10 + 'Iteration time: 2.0\n' * 30)
        measured = measure(Path(directory), log, 'rsl-rl', 8192)
        assert measured['samples_per_second'] == 98304
        assert measured['measured_updates'] == 30
        log.write_text('Iteration time: 2.0\n' * 39)
        with pytest.raises(ValueError, match='Expected 40'):
            measure(Path(directory), log, 'rsl-rl', 8192)


def test_sb3_uses_wall_time_differences_not_cumulative_fps():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        (run / 'tensorboard').mkdir()
        (run / 'tensorboard/events.out.tfevents.test').touch()
        points = [SimpleNamespace(step=(i+1)*8192*24, wall_time=1000+i*2, value=1) for i in range(40)]
        with patch('tensorboard.backend.event_processing.event_accumulator.EventAccumulator') as cls:
            cls.return_value.Scalars.return_value = points
            measured = measure(run, None, 'sb3', 8192)
            assert measured['samples_per_second'] == 98304
            assert measured['measured_updates'] == 30
            points[20].step += 1
            with pytest.raises(ValueError, match='complete rollout'):
                measure(run, None, 'sb3', 8192)


def test_fastest_eligible_count_wins_with_memory_headroom():
    def row(count, speed, memory=20, foreign=None):
        return {'num_envs': count, 'measurement': {'samples_per_second': speed},
                'gpu': {'peak_mib': memory, 'total_mib': 100, 'foreign_pids': foreign or [], 'errors': []}}
    rows = [row(4096, 65), row(8192, 98), row(16384, 93), row(32768, 101, 90), row(65536, 110, foreign=[123])]
    assert choose(rows)['num_envs'] == 8192
    with pytest.raises(ValueError, match='uncontaminated'):
        choose(rows[-2:])


def test_worker_selects_count_then_capacity_gates_without_starting_full():
    import json
    from oh_my_duck.core.paths import project_root
    from oh_my_duck.rl.experiments import worker
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = json.loads((project_root() / 'configs/experiments/representative-throughput.json').read_text())
        plan['runs'] = [next(row for row in plan['runs'] if row['id'] == 'mujoco-sb3-walking')]
        config = root / 'plan.json'
        config.write_text(json.dumps(plan))
        observed = []
        def execute(command, **kwargs):
            if 'train' in command:
                observed.append(command)
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir()
                (destination / 'run.json').write_text(json.dumps({'wall_time_s': 1}))
            return 0
        gpu = {'peak_mib': 10, 'total_mib': 1000, 'foreign_pids': [], 'errors': []}
        measurements = [{'samples_per_second': speed} for speed in (65, 98, 90)]
        argv = ['worker', '--config', str(config), '--run-id', plan['runs'][0]['id'], '--output', str(root/'out'), '--prepare-only']
        with patch('sys.argv', argv), patch.object(worker.subprocess, 'call', side_effect=execute), \
             patch('oh_my_duck.rl.experiments.scaling.GpuSampler') as sampler, \
             patch('oh_my_duck.rl.experiments.scaling.measure', side_effect=measurements):
            sampler.return_value.__enter__.return_value.data = gpu
            assert worker.main() == 0
        report = json.loads((root/'out/result.json').read_text())
        assert report['status'] == 'prepared'
        assert report['spec']['num_envs'] == 8192
        assert report['environment_search_stop'] == 'throughput_regressed'
        assert list(report['stages'])[-2:] == ['capacity', 'capacity-export']
        assert all('full' != Path(c[c.index('--output')+1]).name for c in observed)
        assert observed[-1][observed[-1].index('--num-envs')+1] == '8192'
        assert report['spec']['iterations'] == 50000
