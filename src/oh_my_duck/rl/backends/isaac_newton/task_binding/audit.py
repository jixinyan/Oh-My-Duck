"""Inspect the actual Newton solver against the owned official robot recipe."""
import numpy as np
import torch
import mujoco
import warp as wp
from oh_my_duck.rl.backends.isaac_newton.collision_checks import _points
from oh_my_duck.rl.backends.isaac_newton.task_binding.collisions import reference_model, source_geom


def audit_collisions(env, model):
    reference, actual = reference_model(model), env.sim.mj_model
    labels = [actual.geom(i).name for i in range(actual.ngeom)]
    mapping = {i: source_geom(label, reference) for i, label in enumerate(labels) if '/ground/' not in label}
    ground = next(i for i, label in enumerate(labels) if '/ground/' in label)
    expected = {i for i in range(reference.ngeom) if reference.geom_contype[i] or reference.geom_conaffinity[i]}
    assert set(mapping.values()) == expected and len(mapping) == len(expected)
    directions = np.concatenate((np.eye(3), -np.eye(3), np.random.default_rng(42).normal(size=(64, 3))))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    errors = {}
    for geom, ref in mapping.items():
        error = float(np.max(np.abs((_points(actual, geom) @ directions.T).max(0) - (_points(reference, ref) @ directions.T).max(0))))
        assert error < 1e-4, (labels[geom], error)
        errors[labels[geom]] = error
        assert actual.geom_condim[geom] == reference.geom_condim[ref]
        assert actual.geom_priority[geom] == reference.geom_priority[ref]
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(actual.pair_geom1, actual.pair_geom2)}
    actual_pairs = wp.to_torch(env.sim.wp_model.nxn_pairid).cpu().numpy()[:, 0]
    checked = 0
    for a in range(actual.ngeom):
        for b in range(a + 1, actual.ngeom):
            enabled = int(actual_pairs[checked]) != -2
            ta, aa = (1, 1) if a == ground else (reference.geom_contype[mapping[a]], reference.geom_conaffinity[mapping[a]])
            tb, ab = (1, 1) if b == ground else (reference.geom_contype[mapping[b]], reference.geom_conaffinity[mapping[b]])
            expected_enabled = bool((ta & ab) or (tb & aa))
            if a != ground and b != ground:
                ra, rb = int(reference.geom_bodyid[mapping[a]]), int(reference.geom_bodyid[mapping[b]])
                wa, wb = int(reference.body_weldid[ra]), int(reference.body_weldid[rb])
                pa, pb = int(reference.body_weldid[reference.body_parentid[wa]]), int(reference.body_weldid[reference.body_parentid[wb]])
                same_body = wa == wb
                adjacent = not (reference.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT) and wa != 0 and wb != 0 and (wa == pb or wb == pa)
                excluded = (min(ra, rb) << 16) + max(ra, rb) in reference.exclude_signature
                expected_enabled = expected_enabled and not (same_body or adjacent or excluded)
            assert enabled == expected_enabled, (labels[a], labels[b], enabled)
            checked += 1
    expected_ground = sum(bool(reference.geom_contype[r] & 1 or reference.geom_conaffinity[r] & 1) for r in expected)
    assert actual.npair == expected_ground
    for pair in range(actual.npair):
        a, b = int(actual.pair_geom1[pair]), int(actual.pair_geom2[pair])
        assert ground in (a, b)
        geom = b if a == ground else a
        ref = mapping[geom]
        np.testing.assert_allclose(actual.pair_solref[pair], reference.geom_solref[ref], atol=1e-7)
        np.testing.assert_allclose(actual.pair_solimp[pair], reference.geom_solimp[ref], atol=1e-7)
        assert actual.pair_dim[pair] == reference.geom_condim[ref]
        torch.testing.assert_close(env.sim.model.pair_friction[:, pair], env.sim.model.geom_friction[:, geom][:, [0,0,1,2,2]])
    return {'hull_errors_m': errors, 'mask_pairs_checked': checked, 'ground_pairs': actual.npair}


def audit_randomization(env):
    """Repeating startup DR with the same seed must reproduce, not compound, it."""
    snapshots = []
    for _ in range(2):
        torch.manual_seed(193)
        env.event_manager.apply(mode='startup')
        env.sim.sync_model()
        snapshots.append({name: getattr(env.sim.model, name).clone() for name in env.sim.expanded_fields})
    for name in snapshots[0]:
        torch.testing.assert_close(snapshots[0][name], snapshots[1][name], atol=1e-6, rtol=1e-6, msg=f'Accumulating DR: {name}')
    import warp as wp
    mapping = wp.to_torch(env.sim.solver.mjc_body_to_newton).long()
    valid = mapping >= 0
    native = env.sim.manager._model
    torch.testing.assert_close(wp.to_torch(native.body_mass)[mapping[valid]], env.sim.model.body_mass[valid])
    torch.testing.assert_close(wp.to_torch(native.body_com)[mapping[valid]], env.sim.model.body_ipos[valid])
    return {'non_accumulating_fields': sorted(snapshots[0]), 'native_mass_com_mirror': True}
