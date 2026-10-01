"""Dump official raw MJCF properties on CPU; not training-override parity."""
import json
from pathlib import Path
import sys
import mujoco


def main():
    if len(sys.argv) == 4:
        from oh_my_duck.rl.backends.isaac_newton.asset_names import get_isaac_allcollisions_spec, get_isaac_rollers_spec
        factory = {"robot_allcollisions.xml": get_isaac_allcollisions_spec,
                   "robot_groundcontact_rollers.xml": get_isaac_rollers_spec}[Path(sys.argv[1]).name]
        spec = factory()
        spec.meshdir = str(Path(sys.argv[1]).resolve().parent / "assets")
        model = spec.compile()
        spec.to_file(sys.argv[3])
    else:
        model = mujoco.MjModel.from_xml_path(sys.argv[1])
    bodies = []
    for i in range(1, model.nbody):
        bodies.append({"name": model.body(i).name, "mass": float(model.body_mass[i]),
            "com": model.body_ipos[i].tolist(), "inertia": model.body_inertia[i].tolist(),
            "inertia_quat_wxyz": model.body_iquat[i].tolist()})
    joints = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        d = model.jnt_dofadr[i]
        joints.append({"name": model.joint(i).name, "range": model.jnt_range[i].tolist(),
            "axis": model.jnt_axis[i].tolist(), "armature": float(model.dof_armature[d]),
            "damping": float(model.dof_damping[d]), "frictionloss": float(model.dof_frictionloss[d])})
    Path(sys.argv[2]).write_text(json.dumps({"mujoco": mujoco.__version__, "bodies": bodies, "joints": joints,
        "scope": f"Raw official {Path(sys.argv[1]).name} before training collision and BAM overrides"}, indent=2) + "\n")

if __name__ == "__main__":
    main()
