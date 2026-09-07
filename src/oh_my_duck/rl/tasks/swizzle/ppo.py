# Copyright 2026 Pollen Robotics. Apache-2.0.
# Oh My Duck: task-local native PPO configuration, values preserved.
from .environment import (
    MicroduckRollersRlCfg,
    dataclasses,
)

MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)

