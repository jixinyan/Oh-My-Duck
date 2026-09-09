# Official source and published-policy history audit — 2026-09-08

The current pinned recipe has not accidentally selected the retired Walking
recipe. Dynamic configuration comparison of six train/play variants finds only
8 logging-name changes from historical Velocity2 to current Velocity; StandUp
configuration is identical. This does not establish the unpublished training run
that produced the released policy, nor learned behavior equivalence.

## Source and policy provenance

The historical source is official RL commit `0da91086185a2734ab1bcfdea48cbc6b77cb05bd`
(August 21), compared with pinned `29e887ecfbf5d37144759e5a9f8a176dfb83d547`.
A separate Git archive was extracted for reference; project pins, cached checkouts,
active learners and owned task dependencies are unchanged.

Official [Velocity2 consolidation](https://github.com/pollen-robotics/microduck_rl/commit/4d34d8458a11bb9690cf247d912650057590183e)
records moving the trained/deployed Velocity2 recipe into Velocity. Independent
factory execution covers Flat/Rough Walking and StandUp, each train and play:
only Walking `experiment_name` and `run_name` change from `velocity2` to `velocity`.
This comparison covers resolved configuration and function references, not every
underlying implementation. BAM and the checked core package versions are unchanged:
BAM `62bd8ce`, mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo-Warp 3.8.1,
RSL-RL 5.0.1 and torch 2.9.1.

Byte hashes identify the runtime policy update commits:

| Published file | Official runtime update | SHA-256 |
|---|---|---|
| alpha_walking.onnx | [3954496, August 21](https://github.com/pollen-robotics/microduck/commit/39544966ab901a5434a8ad87338d5d11abae1397) | e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c |
| alpha_stand.onnx | [2430e59, August 24](https://github.com/pollen-robotics/microduck/commit/2430e59ac93367fa36d63cce303c539e00778eeb) | 1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b |

Both match policy-set pin `088524a64e2557dc453256b6071dbb9d23888802` exactly.
Update dates identify published bytes, **not their training run or exact recipe**.
ONNX `run_path` remains `None`. No W&B account was accessed or authenticated.
Official Git objects supplied the audit when web rendering returned cache misses.

## Subsequent CAD export changes

The same corrected CPU/BAM builder compiled historical and current assets.
Total robot mass is identical at 0.73724318 kg. Named joint/body/geom inventory,
contact masks, BAM force limits and the checked DOF fields match. Small inertia
and center-of-mass differences exist; the largest listed inertia change is about
1e-9 kg m² and center-of-mass change 1e-7 m. Quaternion sign reversals represent
the same rotation and are not physical changes.

The first raw mesh comparison was insufficient: vertex ordering and unnamed
geometry can mislead. A second comparison includes **every active geom**, keyed
by body and local ordinal, and transforms vertices to world space at qpos0.
It compares vertex sets without assuming order and samples convex-hull support
along both hulls' face normals and coordinate axes.

| Scene | Active geoms / meshes | Maximum sampled hull-support difference |
|---|---|---|
| Walking | 6 / 5 | 5.45e-9 m |
| Groundcontact | 12 / 11 | 1.75e-6 m |

The largest point-set distance is 0.696 mm, while these collision hull boundaries
are much closer: reordered/retriangulated mesh vertices must not be treated as
a 5 cm physical-model change. This is static evidence against a large CAD collision
change explaining failure; it is **not proof of identical contact trajectories**.
No asset rollback or recipe change follows from this audit.

Evidence: `outputs/baselines/published-source-history-0908-01/` contains the
archive provenance, separately resolved configs, raw differences, compiled-model
comparison, geometry comparison and scripts. `audit-files.json` records hashes.

## Original versus owned checkpoint 1500

All four normalizer-aware exports and corrected CPU batteries completed with
finite traces. These are one-reset-seed progress diagnostics, not final acceptance.

| Task / metric | Original official | Owned |
|---|---|---|
| Walking forward mean, command 0.1 m/s | -0.0002324 m/s | 0.0000100 m/s |
| Walking yaw mean, command 0.5 rad/s | 0.02202 rad/s | 0.02101 rad/s |
| StandUp standing / sitting / prone / supine | pass / fail / fail / fail | pass / pass / fail / fail |

Neither implementation reproduces the full representative behavior at this
checkpoint. Original learners remain active on GPU 7 at 4096 environments each,
using unmodified native PPO/recipes and offline W&B. Later behavior evidence is
still required; the official guidance expects 4000–6000 iterations for gaits and
curriculum-heavy recovery. Owned and Newton learners remain paused as documented.
The second prepared tuning experiment remains deferred, not launched.

The old milestone observer was stopped between assessments: its owned-checkpoint
lookup used pre-resume directories, and waiting for all four learners would block
later original assessments while owned runs are paused. Its outputs and stop
record remain under `matched-growth-0908-01`. Replacement
`official-growth-0908-02` advances each original task independently, locates owned
checkpoints in the correct resume segment when available, records missing matches,
and checks original process identity. Both 1500 assessments are complete. It
changes no training process or recipe. Observer completion means assessment ran;
`cpu_result.status=behavior_failed` remains a failed behavior result.
