"""Load the common policy contract without importing the application or simulator."""
import importlib.util
from .paths import project_root

spec = importlib.util.spec_from_file_location("omd_training_protocol", project_root() / "training/common/protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
JOINT_NAMES = protocol.JOINT_NAMES
HOME = protocol.HOME
OBSERVATION_NAMES = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command", "body_command")


def joint_indices(actual_names):
    names = tuple(actual_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate articulation joint names")
    missing = set(JOINT_NAMES) - set(names)
    if missing:
        raise ValueError(f"Missing policy joints: {sorted(missing)}")
    return tuple(names.index(name) for name in JOINT_NAMES)
