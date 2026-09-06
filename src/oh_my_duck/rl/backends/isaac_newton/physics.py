"""Isolate private Newton solver inspection and source-asset checks."""
from oh_my_duck.rl.backends.isaac_newton.contracts import JOINT_NAMES, joint_indices
from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch


def inspect_solver(env):
    from newton.solvers import SolverMuJoCo
    from isaaclab_newton.physics.newton_manager import NewtonManager
    solver = NewtonManager._solver
    if not isinstance(solver, SolverMuJoCo):
        raise RuntimeError(f"Expected Newton SolverMuJoCo, got {type(solver).__name__}")
    model, data = solver.mjw_model, solver.mjw_data
    robot = env.scene["robot"]
    ids = joint_indices(robot.joint_names)
    actual = env.action_manager.get_term("joint_pos")._joint_names
    if tuple(actual) != JOINT_NAMES:
        raise RuntimeError(f"Resolved action order differs: {actual}")
    fields = {}
    for owner, names in ((model, ("dof_frictionloss", "dof_damping")),
                         (data, ("qfrc_bias", "qfrc_constraint", "qfrc_actuator"))):
        for name in names:
            value = getattr(owner, name, None)
            fields[name] = None if value is None else {"shape": list(value.shape), "strides": list(value.strides)}
    return {"physics": "Newton", "solver": type(solver).__name__, "device": str(data.qpos.device),
        "articulation_joint_names": list(robot.joint_names), "canonical_joint_indices": list(ids),
        "bam_candidate_fields": fields, "bam_implemented": "official_bam" in robot.actuators}


def check_raw_asset(env, reference, armature_override=None):
    import numpy as np
    robot = env.scene["robot"]
    joint_map = {name: i for i, name in enumerate(robot.joint_names)}
    body_map = {name: i for i, name in enumerate(robot.body_names)}
    if set(joint_map) != {j["name"] for j in reference["joints"]}:
        raise AssertionError("Converted joint set differs from raw official MJCF")
    masses = as_torch(robot.data.body_mass)[0].cpu().numpy()
    inertia = as_torch(robot.data.body_inertia)[0].cpu().numpy().reshape(-1, 3, 3)
    com = as_torch(robot.data.body_com_pos_b)[0].cpu().numpy()
    limits = as_torch(robot.data.joint_pos_limits)[0].cpu().numpy()
    armature = as_torch(robot.data.joint_armature)[0].cpu().numpy()
    for ref in reference["joints"]:
        i = joint_map[ref["name"]]
        np.testing.assert_allclose(limits[i], ref["range"], atol=1e-3, rtol=0)
        np.testing.assert_allclose(armature[i], ref["armature"] if armature_override is None else armature_override, atol=1e-6, rtol=0)
    for ref in reference["bodies"]:
        name = ref["name"] + ("_1" if ref["name"] in joint_map else "")
        i = body_map[name]
        np.testing.assert_allclose(masses[i], ref["mass"], rtol=0.01, atol=1e-7)
        np.testing.assert_allclose(np.linalg.eigvalsh(inertia[i]), sorted(ref["inertia"]), rtol=0.01, atol=1e-8)
        np.testing.assert_allclose(com[i], ref["com"], rtol=0, atol=1e-4)
    return {"joint_set": True, "limits": True, "armature": True, "body_mass": True,
        "principal_inertia": True, "body_com": True,
        "pending": ["training collision overrides", "contact parameters", "BAM friction and damping parity", "joint axes"]}
