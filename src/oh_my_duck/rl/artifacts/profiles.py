"""Explicit deployment descriptions for constant-command policy families."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackageProfile:
    name: str
    kind: str
    slot: str | None = None
    duration_s: float | None = None
    entry_pose: str = "standing"


def walking():
    return PackageProfile("walking", "perpetual", slot="walk")


def standup():
    return PackageProfile("standup", "episodic", duration_s=8.0, entry_pose="ground")
