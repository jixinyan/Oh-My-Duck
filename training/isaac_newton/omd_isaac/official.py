"""Shared project-owned BAM configuration, retaining the official motor semantics."""
from copy import deepcopy


def bam_cfg():
    from omd_microduck.robot.microduck_constants import actuators
    return deepcopy(actuators)
