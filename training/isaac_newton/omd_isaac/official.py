"""Load the pinned official implementation without replacing its package sources."""
from copy import deepcopy
import sys
from .paths import project_root


def bam_cfg():
    source = project_root() / ".cache/upstream/microduck_rl/src"
    if not (source / "mjlab_microduck/robot/microduck_constants.py").is_file():
        raise RuntimeError("Pinned official Microduck source missing; run setup")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mjlab_microduck.robot.microduck_constants import actuators
    return deepcopy(actuators)
