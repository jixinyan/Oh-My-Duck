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


class PolicyNetwork:
    """One validated batch-one network with private explicit LSTM memory.

    Failed inference clears memory and neither state is committed until actions
    and both next states are valid. Reset at episode boundaries and whenever a
    caller switches policies, following the official runtime's memory lifetime.
    """

    def __init__(self, path):
        from pathlib import Path
        import numpy as np
        from oh_my_duck.rl.artifacts.publish.manifest import check_onnx

        self.shape = check_onnx(Path(path))
        self.session = cpu_session(path)
        self.state = None
        if self.shape.recurrent:
            self.state = {name: np.zeros(self.shape.state_shape, np.float32)
                          for name in ("h_in", "c_in")}

    def reset(self):
        if self.state is not None:
            for value in self.state.values():
                value.fill(0.0)

    def infer(self, observation):
        import numpy as np

        try:
            obs = np.asarray(observation)
            if obs.shape == (61,):
                obs = obs[None]
            if obs.shape != (1, 61) or obs.dtype != np.float32 or not np.isfinite(obs).all():
                raise ValueError("Policy requires a finite float32 observation of shape [1, 61]")
            feed = {self.shape.input_name: obs}
            names = [self.shape.output_name]
            if self.state is not None:
                feed.update(self.state)
                names += ["h_out", "c_out"]
            values = self.session.run(names, feed)
            action = values[0]
            if action.shape != (1, 14) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError("Policy must produce 14 finite float32 actions")
            if self.state is not None:
                next_states = values[1:]
                if any(value.shape != self.shape.state_shape or value.dtype != np.float32
                       or not np.isfinite(value).all() for value in next_states):
                    raise ValueError("Invalid or nonfinite LSTM output state")
                for name, value in zip(("h_in", "c_in"), next_states, strict=True):
                    np.copyto(self.state[name], value)
            return action
        except Exception:
            self.reset()
            raise

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def get_modelmeta(self):
        return self.session.get_modelmeta()
