# Implementation status

Updated 2026-09-06. The product scope remains the Agentic Microduck Project Design.
Implementation proceeds by domain, with interfaces for later capabilities and
explicit evidence for implemented functionality. Architecture merged into local `main` at `6303ca4`; follow-up validation is on
`feat/rl-pipeline-validation`. No push or public policy upload has occurred.

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
combinations. Four SB3 exports and local packages passed there. After the merge,
all four RSL checkpoints passed export, numerical parity, finite scalar/penalty
checks and local schema-2 packaging using source `6303ca4`. Evidence:
`outputs/postmerge-rsl-export-0906-01/result.json`. The earlier failures remain
preserved; ordinary parity uses elementwise tolerance and extreme stress inputs
use a reported per-action-vector infinity norm tolerance.

MuJoCo EGL pixel readback remains unreliable; explicit OSMesa rendering has
completed Walking task and CPU/BAM replay with valid 720p video. Complete replay,
sim2sim, learned behavior and queued Newton DDP acceptance remain open. These
issues do not block the completed architecture merge.

The latest rendering batch passed 27 lightweight tests and 66 task/SB3 tests. Earlier
installed-runtime checks passed seven Isaac tests and two Newton binding tests.
Local Markdown links and imports contain no obsolete source paths.

Single-GPU work runs directly on the development host; only multi-GPU experiments
use scheduler jobs. W&B remains offline. Current host workloads contend for GPUs,
so observed throughput is not an isolated hardware benchmark.

See [domain refactor evidence](reports/domain-refactor.md),
[RL acceptance](rl-reproduction.md), [architecture](architecture.md), and
[historical implementation log](reports/implementation-history.md).

See [headless rendering evidence](reports/rendering-validation.md) for the explicit software renderer and preserved failed attempts.
