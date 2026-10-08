from pathlib import Path

import numpy as np

from oh_my_duck.rl.artifacts.inference import cpu_session
from oh_my_duck.rl.artifacts.publish.manifest import check_onnx
from oh_my_duck.rl.backends.isaac_newton.contracts import OBSERVATION_NAMES
from oh_my_duck.robotics.microduck.protocol import HOME, JOINT_NAMES


def load_joint_session(path: Path):
    shape = check_onnx(path)
    if shape.recurrent:
        raise ValueError(f"Native joint policy requires a feed-forward API-1 graph: {path}")
    session = cpu_session(path)
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError(f"Joint policy requires one input and one output: {path}")
    if session.get_inputs()[0].type != "tensor(float)" or session.get_outputs()[0].type != "tensor(float)":
        raise ValueError(f"Joint policy requires float32 tensors: {path}")
    metadata = session.get_modelmeta().custom_metadata_map
    if tuple(metadata.get("joint_names", "").split(",")) != JOINT_NAMES:
        raise ValueError(f"Joint policy servo order differs: {path}")
    if tuple(metadata.get("observation_names", "").split(",")) != OBSERVATION_NAMES:
        raise ValueError(f"Joint policy observation order differs: {path}")
    home = np.asarray([float(value) for value in metadata.get("default_joint_pos", "").split(",")])
    if home.shape != (14,) or not np.allclose(home, HOME, atol=5e-4, rtol=0):
        raise ValueError(f"Joint policy HOME differs: {path}")
    if float(metadata.get("action_scale", "nan")) != 1.0:
        raise ValueError(f"Joint policy graph action scale differs: {path}")
    value = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
    if value.shape != (1, 14) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Joint policy CPU inference failed: {path}")
    return session
