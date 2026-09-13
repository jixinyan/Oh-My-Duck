# Implementation status

> 2026-09-13 调度方式更新：后续常规 job 一次执行必要启动检查、完整训练预算及最终评估/视频；检查通过后自动继续，不再常规单独提交短验证 job。当前已在运行的 `omd-rl-causal-0913-01` 仍是原定短验证任务：32 个回放已完成，两任务周期导出已通过数值一致性检查，StandUp 最后容量检查尚在执行。

> 2026-09-13 诊断完成：`omd-rl-diagnose-0912-02` 的 49 个用例全部执行，不能等同于行为通过。Newton RSL 最终起身在两后端 seed=42 均 4/4；CPU/BAM 17 个种子中趴倒 14/17，其余三种姿态各 17/17。其他起身策略仍为 2/4，Walking 未完成验收；Newton Walking 有前进/转向能力，但横向摆动与跨后端前进失败需分开诊断。下一轮补齐只改变推扰强度的配对对照和视频，并验证 Newton 周期导出的真实训练回调；暂不盲目恢复完整训练。下一轮 `omd-rl-causal-0913-01` 已 Running（1 节点 4 张 H800，32 个回放对照后执行两任务短训练门槛）；24 项 CPU 测试与 9 个子测试通过。详见 [因果回放记录](reports/rl-causal-replay-2026-09-13.md)。以下为历史快照。

Updated 2026-09-12. RL iteration resumed by user instruction; single-node,
multi-GPU scheduler jobs are authorized. A 49-case frozen-policy diagnostic
matrix is prepared, including weighted reward traces and 17-seed CPU/BAM checks
of the final Newton StandUp policy. Diagnostic training-stage/push interventions
are explicitly ineligible for standard acceptance. Ten protocol/profile tests and three native export-metadata/callback tests
pass. Newton periodic export metadata indexing is repaired; GPU callback
integration remains pending. Diagnostic job `omd-rl-diagnose-0912-02` is accepted
(1 node, 4 GPUs) and waiting for project quota; attempt 01 failed on a shared lock.
See [current diagnosis](reports/rl-learning-diagnostics-2026-09-12.md). New tasks remain inventory entries
until their own validation, evaluation and packaging gates are implemented.
See [task catalog](rl-task-catalog.md). The pause below is historical.

Updated 2026-09-09. Product scope remains the Agentic Microduck Project Design.

**Training paused by user; zero project RL processes remain.** Five remaining
learners and both live preview workers were stopped with preserved checkpoints,
logs, videos and identity/hash records. No automatic resume is scheduled.
Newton RSL StandUp final passes 4/4 at seed 42 in Newton, MuJoCo and CPU/BAM;
final multi-seed robustness is open. MuJoCo RSL StandUp seed 43 completed with
2/4. Repaired SB3 has stable updates but incomplete learned behavior; Walking
also remains below acceptance. Next priority is causal debugging, not additional
long training. See [current state and debugging handoff](reports/rl-debug-handoff-2026-09-09.md)
for exact saved steps, evidence and the investigation sequence. CLI maturity
reflects partial behavior with training paused. All notes below are historical.

Latest operational update (2026-09-09): four old Walking learners were
intentionally stopped after repeated behavior failures: owned MuJoCo RSL, old
MuJoCo SB3, old Newton SB3, and the original 4096-env MuJoCo RSL control. All
artifacts and checkpoint hashes are preserved with intentional-stop records.
Six learners continue: Newton RSL Walking, the four repaired SB3 runs and the
MuJoCo RSL StandUp seed-43 control (now in full training). Newton RSL StandUp
finished 15000 updates and passed its final native 4/4 pose preview; final
transfer acceptance remains open. The paragraphs below are earlier snapshots.

Latest repair: native SB3 now has separate official critic observations (74D
StandUp / 76D Walking), serialized KL learning-rate feedback, randomized initial
episode phases and matching terminal/normalizer/export/resume paths. 29 tests
and 9 subtests pass. A paired 32-update continuation reduces mean KL from 0.073
to 0.012; this does not prove learned behavior. All four fresh 8192-env runs passed their individual gates and produced full PPO
updates in `sb3-repair-0909-01`, source `bea3eb1`; offline GPU processes verified.
Newton StandUp 13000 passes 12/12 scenarios in each native backend, but CPU/BAM
prone recovery passes only 3/17. No-push Walking 14000 retains forward/turn
response in both native backends; deployment forward motion fails. Details:
[repair and transfer](reports/sb3-critic-transfer-2026-09-09.md).
The SB3 repair milestone is merged and pushed to main `49b56de`; follow-up is
on `feat/rl-transfer-validation`. The complete 14-group frozen-policy battery
confirms MuJoCo StandUp final remains 9/12 in each backend (all supine failures).
Standard Walking with pushes fails acceptance in both backends; no-push results
are diagnostic only. The seed-43 MuJoCo RSL control (`1cb7738`, GPU 1) is running
fresh startup gates before its unchanged 8192-env/15000-update official recipe.
It has a background checkpoint preview worker starting at update 1000.
The following review and operational notes are earlier snapshots.

Latest behavior review (2026-09-09 02:29 UTC): five of the new 8192-env learners
continue, two MuJoCo StandUp runs completed (RSL 3/4, SB3 0/4), and Newton SB3
StandUp was intentionally stopped after ten consecutive 0/4 previews and
persistent excessive KL. Its update-10000 bundle is preserved and hash-verified.
Newton RSL StandUp repeatedly passes its native four-pose battery; final
sim2sim/CPU and multi-seed acceptance remain open. Newton RSL Walking responds
to forward/yaw commands but misses the lateral RMSE gate. Current SB3 critic
input is actor-only 61D versus official RSL's separate 74D input, alongside
native PPO/normalizer differences. Learner equivalence has not been established.
See [behavior and framework review](reports/rl-framework-status-2026-09-09.md).
The following September 8 operational paragraphs are historical snapshots.

Unused-process cleanup completed: all 26 processes in the old paused
`shared-gpu7-0908-01` tree exited, releasing 23.2 GiB on GPU 7. Current eight
learners, original controls and preview workers remain active; retained checkpoint
hashes are unchanged. See the fixed-size training report for the cleanup audit.

Latest user decision supersedes environment sweeps: all eight new combinations
use 8192 environments. Scaling and the old handover were explicitly stopped;
completed matching gates will be reused, with missing gates run independently.
`configs/experiments/representative-8192.json` keeps native PPO and task budgets,
SB3 learning rate 1e-4, and offline W&B. All eight full learners have produced PPO updates and passed startup checks;
each uses 8192 environments. Live process checks confirm offline W&B. Campaign `fixed-8192-0908-01`, source `3070b71`.
See [fixed-size launch evidence](reports/rl-fixed-8192-2026-09-08.md).

Latest scope: train both tasks across both owned backends and both native PPO
frameworks. `measured-env-0908-03` (source `f9fcf80`) is preparing on GPUs
1/6/2/3/4/5: six initial 64-env/5-update training smokes passed; Newton SB3
Walking/StandUp are queued for the next free card. All eight environment counts
remain to be selected from measured PPO throughput with 15% VRAM headroom.
Full training now starts per combination after its own gates pass. The old
global-barrier supervisor is paused while its live preparations continue.
`independent-full-0908-01` (source `392776a`) now adopts each completed
preparation independently; at startup, all new full learners still awaited their
own remaining gates. New campaigns release GPU slots during CPU stages. Original controls continue on 0/7. Their preview controller is temporarily
paused and will automatically resume after isolated calibration. Prior failed
startup artifacts are preserved. See [selection and startup evidence](reports/rl-environment-selection-2026-09-08.md).

Latest user preference: no continuous LLM training polling. A background worker
saves native videos every 1000 PPO updates (24000 control steps per environment)
and at the final checkpoint; unified CPU video/16-reset diagnosis runs after both
original learners end. Earlier intermediate milestone and paused-owned preview
watchers were replaced. Gallery: `outputs/previews/official-periodic-0908-01/index.html`.

Earlier official-first phase: original MuJoCo Walking/StandUp continue. The user
reopened idle GPUs: Walking remains on GPU 7; StandUp migrates from checkpoint
2000 to GPU 0 through native resume; configuration/curriculum audit passed.
GPU 0 later became shared with a foreign process. Diagnostics use GPU 1; a separate
Walking throughput benchmark measured 4096/8192/16384 on GPU 3, with 8192 best
at 98k samples/s. A foreign eight-GPU job confounded the 32768 case and stopped
the sweep before StandUp. Long learners retain 4096. Original StandUp sitting
success remains 15/16 at checkpoint 2500; prone/supine remain 0/16. Eight original
checkpoint-2000 CPU/native StandUp videos are checked. See [resource/progress review](reports/official-training-resource-review-2026-09-08.md).
Owned default Walking (3299) and StandUp (2985) are paused for diagnosis; three
Newton learners remain suspended and three earlier SB3 attempts stopped. Walking
3000 has only 0.98% forward response when native pushes are zeroed, despite 36.4%
in its standard preview; it also nearly stops in nominal native and CPU replay.
The official-guided StandUp angular-penalty diagnostic completed 2500: paired
sitting improves from default 10/16 to 15/16, but prone/supine remain 0/16 each.
All eight final CPU/native videos and local packaging are checked; own-policy
behavior reproduction remains open. The user reaffirmed official-recipe-first
reproduction; the prepared second tuning experiment is deferred and was not launched.

Official history audit confirms six historical/current train/play configurations
differ only in Walking logging names. Published policy bytes are traced to official
runtime commits; their exact training run remains unknown. At checkpoint 1500,
both original/owned Walking nearly stand still in CPU replay; neither StandUp
recovers from prone/supine. A replacement observer fixes resumed checkpoint lookup
and advances original task assessments independently while owned runs are paused.
See [source-history evidence](reports/official-source-history-2026-09-08.md).

The corrected CPU/BAM matrix now completes all eight preserved acceptance policies
with finite 61/14 traces and 20 checked videos; all short policies still fail
behavior. Native previews use scoring v2 (`outputs/previews/shared-gpu7-0908-02`).
See [baseline audit](reports/official-baseline-audit-2026-09-08.md) and
[StandUp diagnosis](reports/standup-recovery-diagnosis-2026-09-08.md), and
[Walking condition/transfer diagnosis](reports/walking-transfer-diagnosis-2026-09-08.md).
SB3 campaign learning-rate routing is tested; a four-combination 1e-4 configuration
is prepared but not launched. Default task recipes and native PPO algorithms remain unchanged.

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
| Published StandUp reference | Official frozen policy passes 64/64 CPU reset samples and 4/4 cases in each native backend; 12 videos checked; own training reproduction remains open |
| Task behavior | All five-iteration policies fail behavior gates; long training and convergence acceptance remain |
| Task replay | All 24 replay contexts completed; 60 videos and finite 61/14 traces checked |
| CPU rehearsal/sim2sim/local packages | All eight corrected CPU/BAM executions revalidated; 20 videos checked; learned behavior remains unverified |
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


## Current video review

`outputs/previews/review-0908-01/index.html` links six existing clips with explicit
labels: default-recipe owned Walking/StandUp, the same Walking policy's CPU
rehearsal, published StandUp in MuJoCo/Newton, and the separate angular-penalty
diagnostic. The gallery does not conflate published-policy replay with own
training success. All six local video targets were checked.
