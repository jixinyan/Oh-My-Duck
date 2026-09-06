import numpy as np
import pytest
from oh_my_duck.rl.artifacts.parity import compare_outputs


def test_stress_cancellation_uses_vector_scale():
    expected = np.ones((32, 14), dtype=np.float32)
    expected[16:, 0] = 100
    actual = expected.copy()
    actual[20, 1] += 1e-4
    report = compare_outputs(expected, actual)
    assert report['ordinary_max_abs_error'] == 0
    assert report['stress_max_normwise_error'] < 1e-5


@pytest.mark.parametrize('row,column,delta', [(0, 1, 1e-3), (20, 1, .01), (20, 0, .01)])
def test_rejects_nominal_or_stress_export_errors(row, column, delta):
    expected = np.ones((32, 14), dtype=np.float32)
    expected[16:, 0] = 100
    actual = expected.copy()
    actual[row, column] += delta
    with pytest.raises(AssertionError):
        compare_outputs(expected, actual)


def test_rejects_nonfinite_native_output():
    expected = np.ones((32, 14), dtype=np.float32)
    actual = expected.copy()
    expected[0, 0] = np.nan
    with pytest.raises(ValueError):
        compare_outputs(expected, actual)
