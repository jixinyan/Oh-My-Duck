from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import imageio.v3 as iio
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.utils.lab_api.math import quat_apply_inverse
from oh_my_duck.rl.evaluation.rendering import read_rgb


def validate_cuda_quaternions():
    quaternion = torch.zeros((64, 4), device="cuda")
    quaternion[:, 0] = 1
    gravity = torch.zeros((64, 3), device="cuda")
    gravity[:, 2] = -1
    # 多次调用实际观测函数，覆盖 TorchScript 在预热之后的 CUDA 编译。
    for _ in range(10):
        actual = quat_apply_inverse(quaternion, gravity)
        torch.testing.assert_close(actual, gravity, rtol=0, atol=0)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), "No allocated CUDA device available to PyTorch"
    wp.init()
    assert wp.is_cuda_available(), "No CUDA device available to Warp"
    validate_cuda_quaternions()
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1" rgba="1 .7 0 1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.3]
        camera.distance = 2
        renderer.update_scene(data, camera=camera)
        iio.imwrite(args.output / "egl.png", read_rgb(renderer))
    packages = {}
    for name in ["torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab", "better-actuator-models", "rsl-rl-lib"]:
        packages[name] = importlib.metadata.version(name)
    info = {"python": platform.python_version(), "platform": platform.platform(), "cwd": str(Path.cwd()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_count": torch.cuda.device_count(), "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch_cuda": torch.version.cuda, "packages": packages, "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "finite_state": True, "torchscript_quaternion": "passed", "egl_render": "egl.png"}
    (args.output / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
