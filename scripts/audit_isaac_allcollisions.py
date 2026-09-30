import json
import argparse
from pathlib import Path
from copy import deepcopy, copy

import mujoco
import numpy as np

from oh_my_duck.robotics.microduck.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG, MICRODUCK_ALLCOLLISIONS_XML,
    name_servo_collision_geoms,
)
from oh_my_duck.rl.backends.isaac_newton.asset_names import get_isaac_allcollisions_cfg


def baseline_spec():
    return name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_cfg = deepcopy(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG)
    baseline_cfg.spec_fn = baseline_spec
    baseline = baseline_cfg.build().compile()
    shared = MICRODUCK_ALLCOLLISIONS_ROBOT_CFG.build().compile()
    scene = get_isaac_allcollisions_cfg().build().compile()
    groups = scene.geom_group.copy()
    snapshot = copy(scene)
    snapshot.geom_group[:] = 5
    if not np.array_equal(scene.geom_group, groups) or np.shares_memory(snapshot.geom_group, scene.geom_group):
        raise RuntimeError("MuJoCo model snapshot shares geometry groups")
    geom_fields = ("geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_priority")
    body_fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature")
    for field in geom_fields + body_fields:
        if not np.array_equal(getattr(baseline, field), getattr(shared, field)):
            raise RuntimeError(f"Shared factory differs: {field}")
    for field in body_fields:
        np.testing.assert_allclose(getattr(scene, field), getattr(shared, field), rtol=0, atol=0)
    differences = []
    for index in range(scene.ngeom):
        changed = {field: {"shared": np.asarray(getattr(shared, field)[index]).tolist(),
                           "scene": np.asarray(getattr(scene, field)[index]).tolist()}
                   for field in geom_fields
                   if not np.array_equal(getattr(shared, field)[index], getattr(scene, field)[index])}
        if changed:
            differences.append({"index": index, "scene_name": scene.geom(index).name, "changes": changed})
    report = {"shared_factory_exact_parity": True, "body_and_armature_exact_parity": True,
              "mujoco_model_copy_independent_geom_group": True,
              "shared_collision_geoms": int(np.count_nonzero(shared.geom_contype | shared.geom_conaffinity)),
              "scene_collision_geoms": int(np.count_nonzero(scene.geom_contype | scene.geom_conaffinity)),
              "scene_configuration": "Raw official allcollisions geometry and material properties with BAM articulation",
              "geometry_differences": differences}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "geometry_differences"}))


if __name__ == "__main__":
    main()
