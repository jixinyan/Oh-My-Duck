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

`oh_my_duck.rl.experiments.throughput` now starts all independent workers
immediately. GPU work takes a process-safe stage lease; CPU BAM rehearsal/video
and local packaging release that lease. Each worker finishes its own smoke,
resume, throughput and capacity/export gates, then enters full training immediately.
There is no all-eight preparation barrier and no repeated preparation for full
training. Actual stage devices are recorded in each worker result. File locks
prevent two managed GPU stages sharing a card; external workloads remain separate.
Full workers subsequently export, package locally, and perform sim2sim/CPU video
acceptance. Periodic checkpoints are saved every 1000 updates plus the final save.
A busy renderer coalesces to the latest save; no assistant reward-monitoring loop runs.

The running legacy `measured-env-0908-03` preparation is retained rather than
restarted. Its old barrier supervisor is paused while workers continue measuring.
`oh_my_duck.rl.experiments.adopt_preparation` can start each ready full learner
from its verified preparation on a free GPU. It validates complete gates, exact
specification and unchanged training/evaluation source. It preserves legacy GPU
reservations until their workers exit, and gives the two already-queued preparatory
workers priority to avoid racing that existing allocator. This transition does
not retroactively change the live legacy worker code. Newly launched campaigns
use stage leases throughout.

Current attempt: `outputs/experiments/measured-env-0908-03`, immutable source
`f9fcf80`, supervisor PID 2232835, launched 2026-09-08 12:57 UTC. Six initial
64-environment / 5-update training smokes passed on GPUs 1/6/2/3/4/5:
MuJoCo RSL Walking/StandUp, MuJoCo SB3 Walking/StandUp, and Newton RSL
Walking/StandUp. Their export/rehearsal/throughput preparation continues. Newton
SB3 Walking/StandUp are queued for the next available GPU. Counts are not yet
selected and the new full-training phase has not started. Original controls
continue on 0/7. No learned-behavior acceptance claim changes.

The coordinator verifies both converted Newton assets before launching workers.
It pauses the original preview controller, lets its in-flight render finish and
release CUDA, then calibrates. Original previews automatically resume after
preparation, including on ordinary failure/interruption. The global full-training barrier is superseded by per-run adoption; selected
counts remain in each preparation result. The successor full campaign records
its own manifest and preview gallery.

Preserved startup attempts:

| Attempt | Observed issue | Resolution |
|---|---|---|
| `measured-env-0908-01`, `f6a0609` | Pausing an in-flight preview retained its CUDA context; isolated benchmark rejected it | Pause only scheduling and drain the existing render |
| `measured-env-0908-02`, `c84f169` | Nested `prepare` directory names collided in native RSL run lookup; Newton fingerprints had changed after metadata edits | Include a campaign-path digest in native tags; validate assets before any training |

Both attempts were explicitly stopped; their partial artifacts are preserved.
Neither reached full training. The stale asset fingerprint came solely from a
policy-inventory addition and an Isaac status label in `configs/upstream.json`.
All robot source bytes, converter/reference/path implementation and dependency
lock matched the accepted conversion, as did every generated artifact hash. The
unchanged USD trees were copied under the new fingerprint with metadata-only
reuse provenance; no new conversion or physical change is claimed. Audit script
and manifests: `outputs/diagnostics/asset-metadata-rekey-0908-01`.

Focused validation: 20 tests plus 9 subtests passed. Tests cover warmup exclusion,
complete SB3 PPO timing, contamination/headroom rejection, selected-count capacity
gates, native learning-rate routing, first-free-GPU reuse, nested run-name isolation
and asset integrity. Changes are committed on `feat/rl-measured-environments`;
no new main merge is claimed. The original Walking benchmark's 8192 result
(~98k samples/s) remains a reference, not the answer for every combination.
