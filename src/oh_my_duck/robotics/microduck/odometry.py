# Copyright 2026 Pollen Robotics. Licensed under Apache-2.0.
# Modified by Oh My Duck: package the generated anchor data behind a stable API.
"""Contact-odometry anchor sets generated from the official Microduck sole mesh.

The points are expressed in each foot site's frame in metres. This module only
loads and validates the generated data; it does not claim that a Python
odometry implementation replaces the deployed runtime's estimator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnchorSet:
    name: str
    left: np.ndarray
    right: np.ndarray

    def foot(self, side: str | int) -> np.ndarray:
        if side in ("left", 0):
            return self.left.copy()
        if side in ("right", 1):
            return self.right.copy()
        raise ValueError("side must be 'left', 'right', 0 or 1")


def _validate_points(name: str, side: str, points: object) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"anchor set {name}/{side} must be a non-empty [N, 3] array")
    if not np.isfinite(array).all():
        raise ValueError(f"anchor set {name}/{side} contains non-finite coordinates")
    return array


@lru_cache(maxsize=1)
def load_anchor_sets(path: str | Path | None = None) -> dict[str, AnchorSet]:
    """Load the generated v15/alpha4/alpha16 anchor sets.

    A custom path is useful for evaluating a newly generated sole mesh without
    mutating the packaged official reference.
    """
    if path is None:
        resource = files("oh_my_duck.robotics.microduck").joinpath("odom_anchor_sets.json")
        source_name = "packaged odom_anchor_sets.json"
        payload = json.loads(resource.read_text())
    else:
        source = Path(path)
        source_name = str(source)
        payload = json.loads(source.read_text())
    if payload.get("units") != "metres, foot-site frame":
        raise ValueError(f"unsupported odometry anchor units in {source_name}")
    sets: dict[str, AnchorSet] = {}
    for name, values in payload.get("sets", {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"anchor set {name} is not an object")
        sets[name] = AnchorSet(
            name=name,
            left=_validate_points(name, "left", values.get("left")),
            right=_validate_points(name, "right", values.get("right")),
        )
    if not sets:
        raise ValueError(f"no anchor sets found in {source_name}")
    return sets


def get_anchor_set(name: str = "alpha16", side: str | int = "left") -> np.ndarray:
    """Return a defensive copy of one named foot's anchor points."""
    sets = load_anchor_sets()
    try:
        return sets[name].foot(side)
    except KeyError as exc:
        raise KeyError(f"unknown odometry anchor set {name!r}; choose from {sorted(sets)}") from exc
