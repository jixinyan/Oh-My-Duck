"""Environment progress accompanying a native SB3 checkpoint."""


def restore_progress(report, native_timesteps):
    if native_timesteps != report['timesteps_after']:
        raise ValueError('Native SB3 checkpoint does not match run metadata')
    if 'env_state' in report:
        steps = report['env_state']['common_step_counter']
        source = 'saved_environment_state'
    else:
        # Old runs started a fresh environment even when the learner resumed.
        # Recover its actual final counter, not an invented cumulative history.
        delta = report['timesteps_after'] - report['timesteps_before']
        count = report['num_envs']
        if count <= 0 or delta < 0 or delta % count:
            raise ValueError('Cannot reconstruct legacy environment progress')
        steps = delta // count
        source = 'legacy_run_delta; earlier legacy resumes restarted curricula'
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError('Invalid curriculum step counter')
    return steps, source
