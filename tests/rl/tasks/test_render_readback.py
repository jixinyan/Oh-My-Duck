import numpy as np
import pytest
from oh_my_duck.rl.evaluation.rendering import read_rgb


class Renderer:
    width = 32
    height = 24

    def render(self, out):
        return out


def test_driver_noop_cannot_expose_uninitialized_memory_as_video():
    with pytest.raises(ValueError, match="empty RGB"):
        read_rgb(Renderer())


def test_valid_readback():
    class Valid(Renderer):
        def render(self, out):
            out[:12] = 255
            return out

    frame = read_rgb(Valid())
    assert frame.shape == (24, 32, 3) and frame.dtype == np.uint8
