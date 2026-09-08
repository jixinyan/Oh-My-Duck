# Measured environment counts for the representative RL matrix

The requested matrix is Walking/StandUp × owned MuJoCo/Isaac-Newton × native
RSL-RL/SB3. Original controls and earlier artifacts remain separate. Full learned
behavior acceptance remains open; successful startup is not gait reproduction.

`configs/experiments/representative-throughput.json` defines a fresh seed-42
campaign with official task rewards, curricula, 24-step rollouts and native PPO.
RSL retains the official recipe. SB3 explicitly uses constant learning rate 1e-4,
based on the existing KL overshoot diagnostic; this is not an official RSL
adaptive learning-rate equivalent. Walking keeps 50000 updates; StandUp 15000.

Per combination, preparation runs 64 environments / 5 updates, normalization-aware
export and reward/penalty checks, CPU BAM video rehearsal, and native resume.
It then measures 4096, 8192, 16384, 32768, 65536 and 131072 environments in ascending
order, stopping if throughput falls more than 5% below the best measured result
or conservative linear VRAM projection would exceed 85% of capacity. Each case
runs 40 updates, discarding 10 warmup updates. Selection is the fastest measured
eligible count, not a proof of a global optimum. No failed or contaminated case
is retried automatically. Native configuration, timings and GPU samples remain
in the output manifest. The selected count passes a fresh capacity/export gate.

RSL uses its native complete iteration timing. SB3 uses differences between
native TensorBoard rollout-boundary timestamps, including PPO update and
collection, rather than startup-biased cumulative FPS. GPU sampling detects
foreign sessions and excludes contaminated measurements. This is a short,
early-curriculum capacity measurement; later contact workload can differ.
Larger environment counts also enlarge PPO batches and total samples for the
same update budget, so faster samples/second does not guarantee faster learning.

`oh_my_duck.rl.experiments.throughput` prepares all combinations before launching
full training from the generated `selected.json`. Limited available GPUs run
independent tasks in per-GPU queues; this is not distributed SB3. Full workers
repeat smoke/capacity gates, then train, export, package locally and run native
sim2sim plus CPU/BAM video acceptance. Periodic previews use checkpoints saved
every 1000 updates plus the final checkpoint; a busy renderer coalesces to the
latest available save. There is no continuous assistant reward monitoring.

At preparation, GPUs 2–5 held other workloads. Original controls remain on 0/7.
The intended calibration allocation is 1/6; the original GPU-1 preview worker
can be identity-checked and paused during measurement, then automatically
resumed. Its checkpoints remain on disk. New full-run previews use GPU 0.

Current state: implementation and focused tests passed; launch evidence will be
recorded separately below. The previous original Walking benchmark preferred
8192 (~98k samples/s) over 4096/16384; other combinations require their own data.

The first launch (`measured-env-0908-01`, source `f6a0609`) passed MuJoCo/RSL
smoke/export (and Walking rehearsal/resume), but Walking calibration was rejected
because a paused pre-existing preview still held a CUDA context. No throughput
result from this attempt is accepted. The attempt was explicitly stopped and all
artifacts retained. The controller now pauses only preview scheduling and drains
its existing child command before calibration. GPUs 2–5 subsequently became free;
a separate six-GPU attempt is planned, preserving original training on 0/7.

Launch evidence: `outputs/experiments/measured-env-0908-02.launch.json` records
supervisor PID 2171344 and immutable source `c84f169`. All six GPUs 1/6/2/3/4/5
were clear of other compute sessions before launch. The six initial preparations
are active; the remaining two combinations use the next available GPU. Existing
original controls remain on 0/7. Environment counts have not yet been selected,
and this new campaign has not yet entered full training. The supervisor runs
all eight preparations, writes `selected.json`, then launches the full matrix
without an assistant monitoring loop. Original previews resume after preparation;
new previews appear at `outputs/experiments/measured-env-0908-02/previews/index.html`.

Focused validation: 14 tests plus 9 subtests passed, including exclusion of
startup/contaminated timing, SB3 complete-update timestamps, selected-count
capacity gating, native learning-rate routing, and reuse of the first free GPU
rather than waiting for a busy assignment. Git source is on
`feat/rl-measured-environments`; no new main merge is claimed.
