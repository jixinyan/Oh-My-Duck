"""Compare the native normalized inference graph with its ONNX artifact on CPU."""
import numpy as np
import torch
import onnxruntime as ort


def verify_runner_export(runner,path):
    native=runner.get_inference_policy(device='cpu').as_onnx(verbose=False).cpu().eval()
    inputs=native.get_dummy_inputs()
    if len(inputs)!=1 or tuple(inputs[0].shape)!=(1,61):
        raise ValueError('A different policy input contract requires its own export audit')
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    observations=np.random.default_rng(42).normal(size=(32,61)).astype(np.float32)
    observations[:,3:6]=[0,0,-1]
    observations[16:]*=100.
    expected=[];actual=[]
    with torch.no_grad():
        for row in observations:
            expected.append(native(torch.from_numpy(row[None])).numpy())
            actual.append(session.run(None,{session.get_inputs()[0].name:row[None]})[0])
    expected,actual=np.concatenate(expected),np.concatenate(actual)
    if expected.shape!=(32,14) or not np.isfinite(actual).all():raise ValueError('Malformed exported actor')
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
    return {'samples':32,'max_abs_error':float(np.max(np.abs(actual-expected))),'status':'passed'}
