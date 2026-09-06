"""Compare actual Newton solver collision masks and hull support to official MJCF."""
import numpy as np
import mujoco
from .official import bam_cfg
from .walk_asset import collider_group


def _points(model, geom):
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    return model.mesh_vert[start:start+count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]


def _key(label):
    if '/left_foot_collision/' in label:
        return ('sole_left', 'ankle_left')
    if '/right_foot_collision/' in label:
        return ('sole_right', 'ankle_right')
    if '/power_support_1/' in label:
        return ('power_support', 'trunk_base')
    if '/upper_leg_left/' in label:
        return ('leg', 'leg')
    if '/upper_leg_right/' in label:
        return ('leg', 'leg_2')
    raise ValueError(f'Unknown official collision mesh: {label}')


def check_walk_collisions(env):
    bam_cfg()  # registers the pinned source path without copying its constants
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
    reference = MICRODUCK_WALK_ROBOT_CFG.build().compile()
    reference_geoms = {}
    for i in range(reference.ngeom):
        if reference.geom_contype[i] or reference.geom_conaffinity[i]:
            reference_geoms[(reference.mesh(int(reference.geom_dataid[i])).name,
                             reference.body(int(reference.geom_bodyid[i])).name)] = i
    solver = env.scene['robot'].actuators['official_bam'].solver
    actual = solver.mj_model
    labels = [actual.geom(i).name for i in range(actual.ngeom)]
    directions = np.concatenate([np.eye(3), -np.eye(3), np.random.default_rng(42).normal(size=(64, 3))])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    checks = []
    seen = set()
    for i, label in enumerate(labels):
        if '/ground/' in label:
            continue
        key = _key(label)
        seen.add(key)
        ref = reference_geoms[key]
        if actual.geom_condim[i] != reference.geom_condim[ref] or actual.geom_priority[i] != reference.geom_priority[ref]:
            raise AssertionError(f'Contact dimension/priority mismatch: {key}')
        np.testing.assert_allclose(actual.geom_friction[i], reference.geom_friction[ref], atol=1e-7)
        expected_support = (_points(reference, ref) @ directions.T).max(axis=0)
        actual_support = (_points(actual, i) @ directions.T).max(axis=0)
        error = float(np.max(np.abs(actual_support - expected_support)))
        if error > 1e-4:
            raise AssertionError(f'Collision hull differs from official by {error:.6g} m: {key}')
        checks.append({'mesh': key, 'support_max_abs_error_m': error})
    if seen != set(reference_geoms):
        raise AssertionError(f'Missing official collision geoms: {set(reference_geoms)-seen}')
    explicit_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(actual.pair_geom1, actual.pair_geom2)}
    pairs = 0
    excluded = set(map(int, actual.exclude_signature))
    for i in range(actual.ngeom):
        for j in range(i+1, actual.ngeom):
            enabled = bool((actual.geom_contype[i] & actual.geom_conaffinity[j]) or
                           (actual.geom_contype[j] & actual.geom_conaffinity[i]))
            body_a, body_b = sorted((int(actual.geom_bodyid[i]), int(actual.geom_bodyid[j])))
            if (body_a << 16) + body_b in excluded:
                enabled = False
            if (i, j) in explicit_pairs:
                enabled = True
            expected = collider_group(labels[i]) == collider_group(labels[j])
            if enabled != expected:
                raise AssertionError(f'Collision mask differs: {labels[i]} / {labels[j]}')
            pairs += 1
    if len(explicit_pairs) != 2:
        raise AssertionError(f'Expected exactly two explicit foot-ground contacts: {explicit_pairs}')
    for pair in range(actual.npair):
        foot = next(g for g in (int(actual.pair_geom1[pair]), int(actual.pair_geom2[pair])) if '/ground/' not in labels[g])
        ref = reference_geoms[_key(labels[foot])]
        f = reference.geom_friction[ref]
        np.testing.assert_allclose(actual.pair_friction[pair], [f[0], f[0], f[1], f[2], f[2]], atol=1e-7)
        np.testing.assert_allclose(actual.pair_solref[pair], reference.geom_solref[ref], atol=1e-7)
        np.testing.assert_allclose(actual.pair_solimp[pair], reference.geom_solimp[ref], atol=1e-7)
        if actual.pair_dim[pair] != reference.geom_condim[ref]:
            raise AssertionError('Explicit foot-ground contact dimension differs')
    return {'status': 'passed', 'explicit_foot_ground_pairs': len(explicit_pairs), 'hulls': checks, 'pairs_checked': pairs,
            'contact_dimension': True, 'priority': True, 'friction': True}
