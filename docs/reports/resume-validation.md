# SB3 curriculum progress on resume

The native mjlab RSL runner saves and restores `env_state.common_step_counter`.
SB3's model ZIP and VecNormalize file do not contain this environment counter.
The adapter now saves it in `run.json` and restores it before SB3 resets the
environment for learning. Native PPO, optimizer and normalization loading remain
unchanged. A mismatched checkpoint timestep count is rejected.

Old run reports lack this field. Their actual final environment counter can be
recovered as `(timesteps_after - timesteps_before) / num_envs`, because those
runs always created a fresh environment. The report explicitly labels this legacy
reconstruction. Earlier legacy resumes restarted their curriculum; reconstructing
the final counter does not undo that historical behavior. New resumes carry the
saved counter forward, independently of the new environment count.

Six regression cases cover saved state, original and resumed legacy runs,
checkpoint identity and invalid counts. The combined task/SB3 CPU suite passes
72 tests. Separate runtime checks follow for both representative tasks and both
simulation backends, including a second resume from a newly saved run.
