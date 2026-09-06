"""Isolated native MuJoCo rendering from the caller's exact simulation state.

The worker imports no learner, Torch or ONNX Runtime. On the development host
EGL readback is unreliable under the installed graphics stack. The worker
can explicitly select OSMesa software rendering. Binary model/state transfer preserves geometry and joint conventions.
"""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def _read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise EOFError("MuJoCo video worker closed its stream")
        chunks.extend(part)
    return bytes(chunks)


class MujocoVideo:
    def __init__(self, model, *, width=1280, height=720, renderer="egl"):
        import mujoco
        from oh_my_duck.infrastructure.headless import rendering_environment

        environment = rendering_environment(renderer)
        self.width, self.height = width, height
        self._temporary = tempfile.TemporaryDirectory(prefix="omd-video-")
        model_path = Path(self._temporary.name) / "scene.mjb"
        mujoco.mj_saveModel(model, str(model_path))
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                __name__,
                "--model",
                str(model_path),
                "--width",
                str(width),
                "--height",
                str(height),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=environment,
        )
        self._closed = False

    def render(self, qpos, qvel, lookat):
        import numpy as np

        state = np.concatenate((qpos, qvel, lookat)).astype("<f8").tobytes()
        stream = self._process.stdin
        stream.write(struct.pack("<I", len(state)))
        stream.write(state)
        stream.flush()
        size = struct.unpack("<I", _read_exact(self._process.stdout, 4))[0]
        if size != self.width * self.height * 3:
            raise ValueError(f"Malformed video worker frame size: {size}")
        return np.frombuffer(_read_exact(self._process.stdout, size), dtype=np.uint8).reshape(
            self.height, self.width, 3
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            code = self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=15)
            raise RuntimeError("MuJoCo video worker did not exit")
        finally:
            self._process.stdout.close()
            self._temporary.cleanup()
        if code:
            raise RuntimeError(f"MuJoCo video worker failed with exit code {code}")


def main():
    from oh_my_duck.infrastructure.headless import configure_egl

    configure_egl()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    from oh_my_duck.rl.evaluation.rendering import read_rgb

    model = mujoco.MjModel.from_binary_path(str(args.model))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 0.85, 135, -20
    try:
        while header := sys.stdin.buffer.read(4):
            if len(header) != 4:
                raise ValueError("Incomplete render state header")
            size = struct.unpack("<I", header)[0]
            if size != (model.nq + model.nv + 3) * 8:
                raise ValueError("Render state does not match the compiled model")
            state = np.frombuffer(_read_exact(sys.stdin.buffer, size), dtype="<f8")
            if not np.isfinite(state).all():
                raise ValueError("Nonfinite render state")
            data.qpos[:] = state[: model.nq]
            data.qvel[:] = state[model.nq : model.nq + model.nv]
            camera.lookat[:] = state[-3:]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = read_rgb(renderer)
            sys.stdout.buffer.write(struct.pack("<I", frame.nbytes))
            sys.stdout.buffer.write(frame.tobytes())
            sys.stdout.buffer.flush()
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
