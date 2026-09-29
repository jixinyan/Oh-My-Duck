import numpy as np
import pytest

import oh_my_duck.robotics.microduck.sim_sensors as sensors


def test_to_uyvy_uses_pair_averaged_chroma_and_rejects_odd_width():
    frame = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    packed = np.frombuffer(sensors.to_uyvy(frame), dtype=np.uint8)
    assert packed.shape == (4,)
    # BT.601 red/blue pair: averaged U/V and distinct luma values.
    assert 169 <= int(packed[0]) <= 171
    assert 75 <= int(packed[1]) <= 77
    assert 180 <= int(packed[2]) <= 182
    assert 28 <= int(packed[3]) <= 30
    with pytest.raises(ValueError, match="even"):
        sensors.to_uyvy(np.zeros((1, 1, 3), dtype=np.uint8))


def test_tof_directions_match_top_left_sensor_convention():
    directions = sensors.tof_directions()
    assert directions.shape == (64, 3)
    assert directions[0, 1] > 0 and directions[0, 2] > 0
    assert directions[7, 1] < 0 and directions[7, 2] > 0
    assert np.all(directions[:8, 2] > 0) and np.all(directions[-8:, 2] < 0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_tof_frame_skips_only_invalid_rays(monkeypatch):
    calls = []

    def fake_ray(model, data, origin, ray, *args):
        calls.append(ray.copy())
        return 1.0

    monkeypatch.setattr(sensors.mujoco, "mj_ray", fake_ray)
    data = type("Data", (), {
        "site_xpos": np.array([[0.0, 0.0, 0.0]]),
        "site_xmat": np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
    })()
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    distance, status = sensors.tof_frame(object(), data, 0, directions=directions,
                                          rng=np.random.default_rng(0))
    assert len(calls) == 1
    assert distance == [1001, 0, 0]
    assert status == [sensors.TOF_STATUS_VALID, sensors.TOF_STATUS_NO_TARGET,
                      sensors.TOF_STATUS_NO_TARGET]
