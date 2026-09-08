# Implementation status

Updated 2026-09-08. The product scope remains the Agentic Microduck Project Design.
Implementation proceeds by domain, with interfaces for later capabilities and
explicit evidence for implemented functionality. Task-family organization, single-GPU pipeline fixes and full-training campaigns
are merged into `main` at `5ef433b`. Submitted training keeps source `1f45996`. No public policy upload has occurred.

Current official-first goal: four MuJoCo learners are active on GPU 7 (original
and owned Walking/StandUp); three Newton learners are suspended, and three earlier
SB3 attempts were stopped. Learned behavior remains unverified. Corrected CPU BAM
A/B is complete for Walking 2000, StandUp 2500 and published alpha; Walking still
fails command response and StandUp passes only standing/sitting. Historical CPU
matrix execution below predates the DOF-load correction and needs revalidation.
Matched checkpoint-250 original/owned CPU comparison also completed: both fail
Walking and pass only standing-start StandUp; early rewards are close, with
long-run equivalence unverified. See the [current baseline audit](reports/official-baseline-audit-2026-09-08.md).

| Component | Current state |
|---|---|
| Source organization | First-party code consolidated under `src/oh_my_duck`; obsolete `training/` packages removed |
| CLI/application | Unified entry point and explicit service composition |
| Core contracts/tool catalog/recording | Tested contracts, tool registry and JSONL event storage |
| Agentic Harness, skills, robot execution | Interfaces; deterministic external-Harness mock remains future work |
| Perception, voice, policy adapters | Interfaces for future implementation; no runtime capability claim |
| MuJoCo RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Isaac/Newton RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Newton physics | Actual solver, canonical state/sensors, precise collisions, BAM cadence, DR and penalties audited |
| Task behavior | All five-iteration policies fail behavior gates; long training and convergence acceptance remain |
| Task replay | All 24 replay contexts completed; 60 videos and finite 61/14 traces checked |
| CPU rehearsal/sim2sim/local packages | Eight historical execution checks completed; corrected CPU load semantics require matrix revalidation; no learned behavior acceptance |
| Multi-GPU | Native RSL MuJoCo DDP previously passed; Newton DDP acceptance pending; old job `a52ff51b` no longer exists in the platform API; no distributed SB3 gradient claim |
| Hardware | Unavailable; all hardware acceptance deferred |

The consolidated batch at `5971dca` passed training and resume for all eight
combinations. Four SB3 exports and local packages passed there. After the merge,
all four RSL checkpoints passed export, numerical parity, finite scalar/penalty
checks and local schema-2 packaging using source `6303ca4`. Evidence:
`outputs/postmerge-rsl-export-0906-01/result.json`. The earlier failures remain
preserved; ordinary parity uses elementwise tolerance and extreme stress inputs
use a reported per-action-vector infinity norm tolerance.

The complete replay matrix used `a0b1f0f`; all eight policies completed both
simulation backends and CPU/BAM rehearsal without runtime errors. Behavior failed
as expected for short smoke checkpoints. Explicit MuJoCo OSMesa video and native
Newton video produced 60 verified 720p clips. SB3 curriculum restoration was then
validated at `b65e8f9` on both tasks/backends, including a second saved-state resume.

Latest tests passed 27 lightweight and 72 task/SB3 cases. Earlier installed-runtime
checks passed seven Isaac and two Newton binding tests. Wheel resources/licenses,
bytecode exclusion, three environment locks and local Markdown links were checked.
See [complete acceptance evidence](reports/rl-pipeline-acceptance.md).

Single-GPU work runs directly on the development host. Following the scheduled
campaign failure, the user also authorized the eight-GPU campaign on the local
H200 host on 2026-09-08; multi-GPU scheduler submission remains available. W&B remains offline. Current host workloads contend for GPUs,
so observed throughput is not an isolated hardware benchmark.

See [domain refactor evidence](reports/domain-refactor.md),
[RL acceptance](rl-reproduction.md), [architecture](architecture.md), and
[historical implementation log](reports/implementation-history.md).

See [headless rendering evidence](reports/rendering-validation.md) for the explicit software renderer and preserved failed attempts.


## Task organization and full-training preparation — 2026-09-07

Task recipes now live in 14 family directories with separate `environment.py` and
`ppo.py`; flat/rough/backlash variants keep the existing registry. All 33 compiled
task catalog entries match the prior semantic inventory (module paths excluded).
Thirty lightweight and 74 task/SB3 tests passed, including native periodic
checkpoint reload and GPU/rank isolation. Full campaigns have a 64-env smoke,
resume and 4096-env capacity gate before the official full iteration budgets.
Long-training submission and worker gates are recorded separately below.


The new campaign worker completed two reduced-size end-to-end gates (RSL-RL and
SB3 Walking) in `outputs/campaign-gate-0907-01`. Every stage completed, including
periodic SB3 bundle resume, final package and both-backend/CPU video. Both reported
`behavior_failed`, with no execution error. These orchestration gates used 64
environments and a two-iteration final segment; the real 4096-environment capacity
gate runs on the allocated training node before long training. Newton binding
checks also passed (two tests). Source/config provenance now includes nested
experiment manifests. Full submission follows from the committed snapshot.


The scheduled attempt `omd-rl-full-0907-01` (platform ID `b2bab260`, original
queue ID `local-662cf40c7dfb`) is **Failed**. Its log is empty, the platform exposes
no log pods, and no campaign manifest exists; the root cause remains unknown.
A separate local attempt `full-local-0908-01` started on eight H200 GPUs using
the same immutable source `1f45996` and training budgets. All eight smoke stages
completed; subsequent gates and full training remain in progress. Failed scheduler
evidence is preserved; local launch metadata is in
`outputs/jobs/omd-rl-local-0908-01/launch.json`.
See [the allocation and evidence record](reports/full-training-2026-09-07.md).


At 2026-09-08 02:08 UTC all eight local runs had entered full training. Early
returns improved relative to initial records; SB3 StandUp shows substantial
regression from its early peak. No learned behavior acceptance is claimed.
See [the measured reward snapshot](reports/reward-trends-2026-09-08.md).


## Triage and one-GPU continuation — 2026-09-08

The two regressing MuJoCo SB3 runs are stopped with artifacts preserved. Six
retained runs resumed native checkpoints on GPU 7 at source `b0fa5fe`; the other
GPUs are released from this campaign. Checkpoint video previews are available at
`outputs/previews/shared-gpu7-0908-01/index.html`. Early RSL task videos show
partial skill progress, with no complete behavior acceptance. A controlled native
SB3 update comparison demonstrates substantially lower KL with smaller learning
rates; stable long-training convergence is still unvalidated. See the
[recovery, diagnosis and video report](reports/rl-recovery-2026-09-08.md).


At the 2026-09-08 04:04 UTC status review, Newton SB3 StandUp also showed
return decline (last 50 logged records 11.95 versus 14.45 previously), elevated
KL (recent values 0.13–0.29) and failure of all four spawn checks on its preceding
preview. It was paused for diagnosis under the user’s standing instruction. Its
latest complete checkpoint at cumulative iteration 1500 is preserved with an
operator-stop record. Five runs continue on GPU 7; this is a triage decision,
not proof that the stopped configuration could never converge.


Official MuJoCo/RSL learned-behavior reproduction remains **unverified**. The
2026-09-08 static audit aligns task/PPO configuration, 228/229 MDP definitions
modulo local imports, BAM and core dependency pins, and the subsequent real-runtime audit verifies the Entity-based StandUp reset
state writes in 15 cases. Matched original/refactor learned-behavior comparison
remains outstanding. See [baseline audit](reports/official-baseline-audit-2026-09-08.md).


Current goal: reproduce official MuJoCo/native RSL learned behavior first, then
match it in Newton and SB3. Isolated pinned-original StandUp PPO smoke and official
export/scalar audit passed; both implementations passed 64-env/5-iteration smoke.
Compiled-model equality, exact reset state/RNG parity and first-reset actor
observation equality are verified. Repeated original runs also show contact-force
and trajectory nondeterminism; long-run behavior is still unverified. Walking
controls and the complete sensor-refresh comparison are in progress. See the
[baseline audit](reports/official-baseline-audit-2026-09-08.md) for evidence and limits.


Walking runtime controls now also match compiled arrays and initial/subset actor
and critic observations; all four original/owned representative PPO smoke runs
completed. A published-policy rehearsal exposed a Walking scoring false positive:
standing almost still met the global RMSE threshold. Scoring v2 now requires signed
motion response in every commanded segment; the preserved published-policy trace
is reclassified `behavior_failed`. This changes acceptance scoring, not training
semantics. Six focused protocol/reset tests pass. See the baseline audit for the
measured commands, responses and remaining interpretation limits.


Official-first resource priority is now active: two owned MuJoCo learners and one
isolated original StandUp control are training on GPU 7. Three live Newton
learners are suspended with SIGSTOP, retaining process/checkpoint state for later
SIGCONT; they are separate from the three earlier stopped SB3 attempts. The
original control passed 4096-env capacity/export and normalized numerical parity
before this status update and is running its 15,000-iteration budget. Baseline
iteration time improved by about 1.5× after suspending extensions. No long-run
behavior success is claimed. Evidence and safe resumption identity records are
linked in the [baseline audit](reports/official-baseline-audit-2026-09-08.md).


CPU/BAM behavioral acceptance requires revalidation: the pinned CPU controller
matched DOF-friction constraints by joint ids, unlike the training Warp path.
The owned CPU adapter now uses actual DOFs while retaining native motor/sag and
friction formulas. Three real-physics/reset tests pass, including independent
Jacobian-force projection and unchanged motor torque. Old traces remain preserved;
corrected policy replay has since completed without restoring the missing behavior (see current baseline audit). This is an evaluation/CPU-load correction,
not evidence that training converged. See the baseline audit.
