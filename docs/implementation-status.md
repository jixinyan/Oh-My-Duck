# Implementation status

Updated 2026-09-07. The product scope remains the Agentic Microduck Project Design.
Implementation proceeds by domain, with interfaces for later capabilities and
explicit evidence for implemented functionality. Architecture merged into local `main` at `6303ca4`; follow-up validation is on
`feat/rl-pipeline-validation`. No public policy upload has occurred.

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
| CPU rehearsal/sim2sim/local packages | All eight policies exported/packaged and replayed in both backends plus CPU/BAM |
| Multi-GPU | Native RSL MuJoCo DDP previously passed; Newton DDP running (job `a52ff51b`), acceptance pending; no distributed SB3 gradient claim |
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

Single-GPU work runs directly on the development host; only multi-GPU experiments
use scheduler jobs. W&B remains offline. Current host workloads contend for GPUs,
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


Full training has been submitted as `omd-rl-full-0907-01`, queue ID
`local-662cf40c7dfb`, source `1f45996`; latest submission state is Pending.
The single eight-GPU node runs all eight task/backend/framework combinations.
See [the allocation and evidence record](reports/full-training-2026-09-07.md).
