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
