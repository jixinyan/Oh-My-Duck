"""Process-safe GPU leases scoped to GPU work, not CPU rehearsal/video stages."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

_active = None


@contextmanager
def gpu_lease():
    global _active
    directory = os.environ.get('OMD_GPU_POOL')
    if not directory or _active is not None:
        yield _active or os.environ.get('CUDA_VISIBLE_DEVICES')
        return
    devices = os.environ['OMD_GPU_POOL_DEVICES'].split(',')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    stream = None
    try:
        while stream is None:
            for device in devices:
                candidate = (root / f'{device}.lock').open('a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                stream = candidate
                _active = device
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                break
            if stream is None:
                time.sleep(0.2)
        yield _active
    finally:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
        _active = None
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
