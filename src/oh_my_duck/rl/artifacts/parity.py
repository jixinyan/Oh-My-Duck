"""Compare the native normalized inference graph with its ONNX artifact on CPU."""

import numpy as np
import torch
from oh_my_duck.rl.artifacts.inference import cpu_session


def compare_outputs(expected, actual, ordinary_samples=16):
    """Elementwise nominal gate; normwise FP32 gate for extreme stress inputs.

    An unbounded RSL actor can produce hundreds at 100x input scale. Relative
    error of a nearly cancelled output then exaggerates GEMM rounding. The
    stress gate uses each action vector's infinity norm, retaining the same
    1e-5 relative tolerance. Nominal samples retain the stricter element gate.
    """
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Malformed exported actor")
    np.testing.assert_allclose(actual[:ordinary_samples], expected[:ordinary_samples], atol=2e-5, rtol=1e-5)
    errors = np.abs(actual - expected)
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=1, keepdims=True))
    if np.any(errors[ordinary_samples:] > (2e-5 + 1e-5 * scale[ordinary_samples:])):
        raise AssertionError("Export stress parity exceeds the FP32 normwise tolerance")
    return {
        "samples": len(expected), "max_abs_error": float(errors.max()),
        "ordinary_max_abs_error": float(errors[:ordinary_samples].max()),
        "stress_max_normwise_error": float((errors[ordinary_samples:] / scale[ordinary_samples:]).max()),
        "ordinary_tolerance": {"atol": 2e-5, "rtol": 1e-5, "mode": "elementwise"},
        "stress_tolerance": {"atol": 2e-5, "rtol": 1e-5, "mode": "per-vector infinity norm"},
        "status": "passed",
    }


def verify_runner_export(runner, path):
    native = runner.get_inference_policy(device="cpu").as_onnx(verbose=False).cpu().eval()
    inputs = native.get_dummy_inputs()
    if len(inputs) != 1 or tuple(inputs[0].shape) != (1, 61):
        raise ValueError("A different policy input contract requires its own export audit")
    session = cpu_session(str(path), providers=["CPUExecutionProvider"])
    observations = np.random.default_rng(42).normal(size=(32, 61)).astype(np.float32)
    observations[:, 3:6] = [0, 0, -1]
    observations[16:] *= 100.0
    expected = []
    actual = []
    with torch.no_grad():
        for row in observations:
            expected.append(native(torch.from_numpy(row[None])).numpy())
            actual.append(session.run(None, {session.get_inputs()[0].name: row[None]})[0])
    expected, actual = np.concatenate(expected), np.concatenate(actual)
    if expected.shape != (32, 14) or not np.isfinite(actual).all():
        raise ValueError("Malformed exported actor")
    return compare_outputs(expected, actual)
