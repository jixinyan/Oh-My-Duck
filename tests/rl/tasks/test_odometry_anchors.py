import numpy as np
import pytest

from oh_my_duck.robotics.microduck.odometry import get_anchor_set, load_anchor_sets


def test_packaged_anchor_sets_are_finite_and_mirrored_per_foot():
    sets = load_anchor_sets()
    assert {"v15", "alpha4", "alpha16"} <= set(sets)
    assert sets["v15"].left.shape == (4, 3)
    assert sets["alpha4"].left.shape == (4, 3)
    assert sets["alpha16"].left.shape == (16, 3)
    for anchor in sets.values():
        assert np.isfinite(anchor.left).all() and np.isfinite(anchor.right).all()


def test_anchor_selection_returns_a_copy_and_rejects_unknown_names():
    points = get_anchor_set("alpha16", "right")
    original = points.copy()
    points[0, 0] = 999.0
    np.testing.assert_array_equal(get_anchor_set("alpha16", "right"), original)
    with pytest.raises(KeyError, match="unknown"):
        get_anchor_set("missing")
    with pytest.raises(ValueError, match="side"):
        get_anchor_set("v15", "front")
