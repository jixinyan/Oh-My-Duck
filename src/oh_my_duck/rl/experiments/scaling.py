"""Measure complete native PPO iterations and select an explicit environment count."""
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading


def measure(run, log, framework, count, warmup=10, updates=40):
    """Exclude startup; SB3 event boundaries include collection AND native PPO."""
    if framework == 'rsl-rl':
        text = re.sub(r'\x1b\[[0-9;]*m', '', Path(log).read_text(errors='replace'))
        durations = [float(x) for x in re.findall(r'Iteration time:\s+([\d.]+)', text)]
        if len(durations) != updates:
            raise ValueError(f'Expected {updates} native iteration timings, got {len(durations)}')
        durations = durations[warmup:]
        method = 'native_iteration_time'
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        files = list((Path(run) / 'tensorboard').rglob('events.out.tfevents.*'))
        if len(files) != 1:
            raise ValueError('Expected one fresh native SB3 event stream')
        events = EventAccumulator(str(files[0]), size_guidance={'scalars': 0})
        events.Reload()
        points = events.Scalars('time/fps')
        if len(points) != updates:
            raise ValueError(f'Expected {updates} rollout boundaries, got {len(points)}')
        pairs = list(zip(points, points[1:]))[warmup-1:]
        if any(b.step - a.step != count * 24 for a, b in pairs):
            raise ValueError('SB3 benchmark must advance by one complete rollout per boundary')
        durations = [b.wall_time - a.wall_time for a, b in pairs]
        method = 'native_tensorboard_boundary_wall_time'
    if len(durations) != updates - warmup or not all(math.isfinite(x) and x > 0 for x in durations):
        raise ValueError('Incomplete or invalid steady-state timings')
    return {'method': method, 'warmup_updates': warmup, 'measured_updates': len(durations),
            'iteration_mean_s': statistics.mean(durations),
            'iteration_median_s': statistics.median(durations),
            'samples_per_second': count * 24 * len(durations) / sum(durations)}


def choose(rows, memory_fraction=0.85):
    eligible = [r for r in rows if not r['gpu']['foreign_pids'] and not r['gpu']['errors']
                and r['gpu']['peak_mib'] <= r['gpu']['total_mib'] * memory_fraction]
    if not eligible:
        raise ValueError('No uncontaminated measurement with sufficient VRAM headroom')
    return max(eligible, key=lambda r: r['measurement']['samples_per_second'])


class GpuSampler:
    """Detect competing processes; sample VRAM, never terminate another GPU user."""
    def __init__(self):
        self.device = os.environ['CUDA_VISIBLE_DEVICES']
        if not self.device.isdecimal():
            raise ValueError('Benchmark requires one explicit numeric GPU')
        self.data = {'device': self.device, 'peak_mib': 0, 'total_mib': 0,
                     'foreign_pids': [], 'errors': []}
        self.stop = threading.Event()

    def sample(self):
        try:
            line = subprocess.check_output(['nvidia-smi', '--id=' + self.device,
                '--query-gpu=uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
            uuid, total, used = [x.strip() for x in line.strip().split(',')]
            self.data['total_mib'] = int(total)
            self.data['peak_mib'] = max(self.data['peak_mib'], int(used))
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader'], text=True)
            foreign = set(self.data['foreign_pids'])
            for line in apps.splitlines():
                gpu, pid = [x.strip() for x in line.split(',')]
                if gpu == uuid:
                    try:
                        if os.getsid(int(pid)) != os.getsid(0):
                            foreign.add(int(pid))
                    except ProcessLookupError:
                        pass
            self.data['foreign_pids'] = sorted(foreign)
        except Exception as error:
            self.data['errors'].append(repr(error))

    def __enter__(self):
        self.sample()
        if self.data['foreign_pids'] or self.data['errors']:
            raise RuntimeError(f'Benchmark GPU is not isolated: {self.data}')
        def loop():
            while not self.stop.wait(5):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop.set()
        self.thread.join()
        self.sample()
