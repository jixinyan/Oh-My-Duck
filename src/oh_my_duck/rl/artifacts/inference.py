"""CPU inference sessions for small joint policies and numerical export audits.

Use an explicit single CPU thread for batch-one policy inference. This controls
host thread overhead; EGL video correctness is validated separately and is not
guaranteed by the inference thread setting.
"""


def cpu_session(path, **kwargs):
    import onnxruntime as ort

    if kwargs.get("providers", ["CPUExecutionProvider"]) != ["CPUExecutionProvider"]:
        raise ValueError("This policy inference path explicitly requires the CPU provider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
