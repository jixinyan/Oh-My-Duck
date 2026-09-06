"""Load the common policy contract without importing the application or simulator."""
from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES, HOME
OBSERVATION_NAMES = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command", "body_command")


def joint_indices(actual_names):
    names = tuple(actual_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate articulation joint names")
    missing = set(JOINT_NAMES) - set(names)
    if missing:
        raise ValueError(f"Missing policy joints: {sorted(missing)}")
    return tuple(names.index(name) for name in JOINT_NAMES)
