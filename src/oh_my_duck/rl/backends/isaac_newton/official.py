"""Shared project-owned BAM configuration, retaining the official motor semantics."""
from copy import deepcopy


def bam_cfg():
    from oh_my_duck.robotics.microduck.microduck_constants import actuators
    return deepcopy(actuators)
