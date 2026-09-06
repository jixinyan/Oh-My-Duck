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
72 tests. Runtime checks at `b65e8f9` passed both representative tasks and both
simulation backends: restored counter 120, saved counter 240, followed by fresh
normalized exports and local packages. A second MuJoCo StandUp resume restored
240 from the new manifest and saved 360. Every first resume recorded 768 timeout
snapshots. Evidence is in `outputs/sb3-curriculum-resume-0906-01/`.
The initial training/replay artifacts remain valid; earlier resume evidence did
not establish curriculum continuity.
