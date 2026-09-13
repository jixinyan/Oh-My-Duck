"""Official export metadata for either runtime's canonical Microduck contract."""
from types import SimpleNamespace
from mjlab.rl.exporter_utils import get_base_metadata


def policy_metadata(env, run_path):
    robot = env.scene['robot']
    reference = getattr(robot, 'reference_model', None)
    if reference is None:
        return get_base_metadata(env, run_path)
    # Newton applies BAM forces through joint DOFs, so the solver's MuJoCo
    # representation contains no native actuators. The official exporter indexes
    # actuator arrays using IDs from robot.spec (the canonical MJCF), which must
    # therefore be read from that same compiled reference. This is metadata only:
    # never swap or mutate the running physics model.
    view = SimpleNamespace(scene=env.scene, action_manager=env.action_manager,
                           command_manager=env.command_manager,
                           observation_manager=env.observation_manager,
                           sim=SimpleNamespace(mj_model=reference))
    metadata = get_base_metadata(view, run_path)
    metadata.update(actuator_metadata_source='official_nominal_mjcf',
                    simulation_backend='isaac-newton', actuator_model='BAM_XL330_M6')
    return metadata


def verify_periodic_export(periodic, reference):
    """Require the actual Newton callback artifact to match official run_export."""
    import hashlib
    from pathlib import Path
    import numpy as np
    from oh_my_duck.rl.artifacts.inference import cpu_session
    from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES
    sessions = [cpu_session(str(path), providers=['CPUExecutionProvider']) for path in (periodic, reference)]
    metadata = sessions[0].get_modelmeta().custom_metadata_map
    if (metadata.get('simulation_backend') != 'isaac-newton'
            or metadata.get('actuator_model') != 'BAM_XL330_M6'
            or metadata.get('actuator_metadata_source') != 'official_nominal_mjcf'
            or tuple(metadata.get('joint_names', '').split(',')) != JOINT_NAMES):
        raise AssertionError('Missing canonical Newton BAM callback metadata')
    for session in sessions:
        assert session.get_inputs()[0].shape == [1, 61]
        assert session.get_outputs()[0].shape == [1, 14]
    rng = np.random.default_rng(42)
    samples = [np.zeros((1, 61), dtype=np.float32)] + [rng.normal(size=(1, 61)).astype(np.float32) * scale for scale in [1.] * 32 + [3.] * 32]
    error = 0.
    for obs in samples:
        values = [session.run(None, {session.get_inputs()[0].name: obs})[0] for session in sessions]
        if not all(np.isfinite(value).all() for value in values):
            raise AssertionError('Nonfinite periodic export output')
        np.testing.assert_allclose(*values, atol=1e-6, rtol=1e-5)
        error = max(error, float(np.max(np.abs(values[0] - values[1]))))
    return {'status': 'passed', 'samples': len(samples), 'max_abs_error': error,
            'periodic_sha256': hashlib.sha256(Path(periodic).read_bytes()).hexdigest(),
            'reference_sha256': hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
            'scope': 'Actual native save callback versus normalizer-aware official run_export; not learned behavior'}
