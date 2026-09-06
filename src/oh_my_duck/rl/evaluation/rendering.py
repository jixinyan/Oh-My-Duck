"""Checked pixel readback for native MuJoCo renderers."""

import numpy as np


def read_rgb(renderer):
    # Native Renderer defaults to np.empty. If driver readback is a no-op,
    # arbitrary heap contents can pass a nonuniform-image check. Zero first.
    pixels = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
    frame = renderer.render(out=pixels)
    if frame.shape != pixels.shape or not np.isfinite(frame).all() or np.ptp(frame) < 1:
        raise ValueError("MuJoCo produced an empty RGB frame; check the EGL vendor configuration")
    return frame
