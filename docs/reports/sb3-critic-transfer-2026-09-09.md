# SB3 critic repair and RSL cross-simulator diagnosis

Follow-up to the [framework review](rl-framework-status-2026-09-09.md).
User requested SB3 repair, diagnosis of the weaker MuJoCo RSL run, and capability
validation of the Newton RSL policies including MuJoCo transfer.

## Critic interface implementation

Fresh SB3 training now defaults to separate official actor/critic observation
groups. Native SB3 PPO uses its DictRolloutBuffer and its existing timeout
bootstrapping. Independent VecNormalize statistics are saved for each group.
The policy keeps 61 actor inputs, 14 outputs and task-local network widths;
critic observations cannot enter actor hidden layers or its exported graph.
Both terminal groups are sampled before reset without advancing live delay
buffers or consuming the training RNG. Export still uses official run_export
and the runner export method, with actor normalization baked in.

`--critic-observations actor` preserves the former input for controlled studies.
Resume infers the recorded layout (legacy artifacts imply actor-only) and rejects
changing layout while restoring a checkpoint. Old exports remain supported.
PPO optimizer, value loss and learning-rate defaults are unchanged in this
isolated interface change. This does not claim that critic repair alone fixes
learning behavior.

CPU checks: 14 SB3 tests passed, including privileged-input/gradient isolation,
native DictRolloutBuffer training and normalized terminal-critic bootstrapping,
model/normalizer save-reload, asymmetric actor graph parity, legacy graph parity,
and real observation-manager delay/history preservation. Required native
64-env/5-update, export and CPU/BAM video gates are pending.

## Frozen-policy transfer experiment

`outputs/diagnostics/newton-transfer-0909-02` uses immutable source `668b661`:
Newton StandUp checkpoint 13000 and Walking 14000, plus final MuJoCo RSL StandUp.
Native backends test recovery at seeds 42/100/101; CPU/BAM tests recovery at 42
and 100–115. Seed 42 includes videos. The same ONNX bytes are checked in each
backend. This avoids conflating simulator transfer with another training run.

Attempt 01 failed before any rollout because the diagnostic driver globally
selected OSMesa before loading its optional library. Attempt 02 keeps EGL in
the parent and selects OSMesa only in the existing isolated MuJoCo video worker;
both attempts and their logs are retained. No training physics changed.
Results and remaining causal uncertainty will be recorded after completion.

## Native SB3 learning-rate feedback

A separate serializable callable schedule and rollout-boundary callback now
reduce/increase the learning rate from SB3's preceding logged approximate KL.
The native PPO update and early-stop implementation remain unchanged. Bounds
1e-5–1e-2 and factor 1.5 match the official RSL controller's bounds, but timing
and KL estimator differ: this is not numerically identical RSL PPO. Fresh runs
default to adaptive; resume preserves the saved mode. Explicit constant mode
supports matched comparisons, including legacy checkpoints. Both schedule state
and native optimizer learning rate survive save/reload in the tests.

16 SB3 CPU tests pass. A real checkpoint comparison of constant versus adaptive
updates is still required before claiming the observed KL instability is fixed.
The critic-only gates run from source `c897d10` and retain constant learning rate;
the new controller is a separate change, not silently added to those runs.

Transfer evidence so far: Newton StandUp 13000 in CPU/BAM passes standing,
sitting and supine at all 17 tested base seeds; prone passes only 3/17. Native
4/4 preview success is therefore not full transfer success. Native paired tests
use each backend's registered interpreter in attempt 03; attempt 02 incorrectly
attempted MuJoCo inside Newton's isolated dependency environment and failed
before a native rollout. CPU attempt 02 remains valid and continues independently.

## Initial episode phases and RSL diagnosis

The official RSL entry point calls `learn(init_at_random_ep_len=True)`. Fresh
SB3 runs now use the same one-time uniform episode-length initialization, so
thousands of worlds do not all time out on the same control step. Ordinary
subsequent resets still reset their counters normally. Native SB3 PPO and vector
stepping remain intact; this is not a decoupled actor implementation. Explicit
`--initial-episode-phase synchronized` supports comparison; legacy resume keeps
its previous setting. All three runtime choices are recorded in checkpoints.
17 SB3 CPU tests passed, including deterministic randomized initial phases.

The RSL source-668b661 reconstruction at 8192 environments/seed 42 finds exact
initial actor tensor hashes across backends and identical recorded solver
options. Saved full-run environment YAML is identical; agent YAML differs only
in run_name. CUDA RNG state before runner construction differs despite the same
seed; CPU RNG and initial actor weights match. This establishes different
exploration streams, not proof that randomness alone explains the final gap.
Evidence: `outputs/diagnostics/rsl-backend-initialization-0909-01`.

Newton StandUp 13000 transfers to native MuJoCo at seeds 42/100/101 with 4/4
successes each. MuJoCo can execute the complete behavior; its own final policy's
missing supine skill is a learning outcome, not an inability of the simulator to
represent it. CPU/BAM still exposes a prone-transfer gap. No official rewards,
curricula, robot parameters or RSL defaults have been changed.

## Verified update comparison and next full runs

The fixed-checkpoint comparison (`outputs/diagnostics/sb3-kl-feedback-0909-01`)
completed with identical old update-10000 model/normalizer, 8192 environments,
seed 42, synchronized initial phases and actor-only critic in both arms. Only
KL feedback differs; initial learning rate is 1e-4. Each arm executed 32 native
PPO rollouts. SB3 logs the preceding update, yielding 31 displayed KL samples.

| Metric | Constant | Adaptive |
|---|---:|---:|
| Mean logged KL | 0.07300 | 0.01196 |
| Last 16 mean KL | 0.05339 | 0.00858 |
| Maximum logged KL | 0.19825 | 0.03440 |
| Early-stop messages / 32 rollouts | 32 | 16 |
| Final learning rate | 1e-4 | 2.963e-5 |

This is evidence for improved update stability, not learned-skill recovery.
The critic-only four-combination gates completed from `c897d10`: all native
64-env/5-update runs, resumes, official-route normalized exports and CPU/BAM
videos executed. Smoke behavior failures are retained and expected.

`configs/experiments/sb3-official-critic.json` makes critic layout, learning-rate
mode and initial episode phase explicit at every worker stage. The new campaign
will rerun all gates from the combined committed implementation before fresh
full training. No old gate is reused across changed policy/model source. Policy
model sources are now included in preparation invalidation. Regression checks:
29 tests and 9 subtests passed for SB3, campaign options, recovery and scheduling.
