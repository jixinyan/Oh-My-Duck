# Implementation status

Updated 2026-09-06. The product scope remains the Agentic Microduck Project Design.
Implementation proceeds by domain, with interfaces for later capabilities and
explicit evidence for implemented functionality. Work is local on
`feat/domain-architecture`; no push or public policy upload has occurred.

| Component | Current state |
|---|---|
| Source organization | First-party code consolidated under `src/oh_my_duck`; obsolete `training/` packages removed |
| CLI/application | Unified entry point and explicit service composition |
| Core contracts/tool catalog/recording | Tested contracts, tool registry and JSONL event storage |
| Agentic Harness, skills, robot execution | Interfaces; deterministic external-Harness mock remains future work |
| Perception, voice, policy adapters | Interfaces for future implementation; no runtime capability claim |
| MuJoCo RL | Both representative tasks × both native PPO frameworks have smoke/resume/export evidence |
| Isaac/Newton RL | Both representative tasks × both native PPO frameworks have smoke/resume/export evidence |
| Newton physics | Actual solver, canonical state/sensors, precise collisions, BAM cadence, DR and penalties audited |
| Task behavior | Short smoke policies fail learned-behavior gates; longer training awaits consolidated acceptance |
| Task replay | Shared protocol API, manual resets and 720p video implemented; Newton four-pose replay executed |
| CPU rehearsal/sim2sim/local packages | Unified task interfaces implemented; consolidated runtime validation in progress |
| Multi-GPU | Native RSL MuJoCo DDP previously passed; Newton DDP queued; no distributed SB3 gradient claim |
| Hardware | Unavailable; all hardware acceptance deferred |

The consolidated batch at `5971dca` passed training and resume for all eight
combinations. All four SB3 exports/local packages and the MuJoCo Walking RSL
export/package passed. Three RSL exports hit the extreme-input numerical gate;
normal inputs passed. The revised gate retains elementwise nominal checks and
reports normwise stress error separately; runtime revalidation is pending.
MuJoCo EGL video remains intermittent, including in CPU rehearsal. These are
open validation issues, not evidence of accepted learned policies. The architecture
is ready to merge locally; validation continues on a separate feature branch.

The most recent CPU batch passed 24 lightweight tests and 59 task/SB3 semantic
tests. Installed-runtime Isaac/sensor checks and the final end-to-end batch follow.

Single-GPU work runs directly on the development host; only multi-GPU experiments
use scheduler jobs. W&B remains offline. Current host workloads contend for GPUs,
so observed throughput is not an isolated hardware benchmark.

See [domain refactor evidence](reports/domain-refactor.md),
[RL acceptance](rl-reproduction.md), [architecture](architecture.md), and
[historical implementation log](reports/implementation-history.md).
