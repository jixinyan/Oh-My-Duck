# Development conventions

## Boundaries and entry points

`omd.py` / `oh_my_duck.cli` is the public CLI. All first-party implementation is
under `src/oh_my_duck/`, grouped by project capability; see [architecture.md](architecture.md).
`infrastructure/` owns setup, process dispatch and scheduler submission and imports
no simulator at startup. `rl/` owns task recipes, MDP, learners, export and evaluation;
`robotics/microduck/` owns models, assets, motor behavior and robot conventions.

Task factories and policy configuration are editable project source. Native RSL-RL
and SB3 PPO remain dependencies. Microduck task code is never loaded from the
upstream cache. `environments/{mujoco,isaac-newton,isaac-assets}` holds dependency
manifests and locks; run their setup through `omd setup`.

The Isaac backend requires Newton. Unsupported task bindings fail explicitly.
Keep training dependencies out of agentic, voice, recording and CLI imports.

## Files and provenance

- Commit code, small reproducible configs, documentation, audit inventories and compact measured reports.
- Ignore upstream checkouts, Python environments, package caches, credentials, checkpoints, videos and raw trajectories. Obtain upstreams from the checked-in revision manifest.
- Each job runs from a detached Git worktree of its committed revision under `.job-sources/`; source imports and configuration point at that snapshot. Commit pending changes before submission. Environments, pristine upstream dependencies and output directories are shared through explicit links. Existing snapshots are never edited.
- Each job has a unique name and immutable output directory. Save source revision, dirty state, actual invocation, dependency versions, source/model hashes and results alongside outputs.
- Worker commands use shared absolute paths. `/tmp` is for disposable research, not distributed-job dependencies.
- Large evidence stays under `outputs/` / `logs/`; link it from compact reports with hashes and job identifiers. A machine-local output path is not a portable published artifact.
- Keep upstream code and asset notices separate. The maintained Microduck source meshes are package data with preserved upstream ancestry. Generated USD and downloaded policy weights remain ignored.

## Version control

Local `main` is initialized with `origin=https://github.com/jixinyan/Oh-My-Duck.git`. Remote inspection failed with the current credentials; no remote history is assumed and no force-push is allowed. When access becomes available, inspect/fetch remote history and reconcile it before pushing local commits.

Create a branch before an independent feature, fix or substantial investigation: `feat/<topic>`, `fix/<topic>` or `docs/<topic>`. Keep `main` at a reviewed baseline. A completed branch can be merged locally after its relevant checks pass; inspect remote history before any eventual push.

Commit at coherent, reviewable checkpoints: interfaces/scaffolding, working behavior with its relevant checks, and measured validation evidence. Do not accumulate an entire feature into one final commit or make empty commits merely to increase frequency. Keep documentation describing a behavior in the same commit as that behavior. Record incomplete implementations explicitly rather than implying validation.

Before committing, inspect the staged diff and ensure no ignored artifacts or credentials entered the index. After committing, check `git status` and `git log`, and report the branch, commit IDs and whether they are local or pushed. Do not change the user's Git identity.

History at the start of the Isaac reference review: local `main` already contained `bca0a90` (whole-project framework). The reference review uses `docs/isaac-newton-reference-review`; no commits have been pushed. Local commits are visible through `git log --all --oneline --decorate`, independently of GitHub access.

## Progress

Update the original design and execution documents when scope, module responsibilities or milestone status changes. Distinguish implemented, executed and validated. Record failed experiments, including job IDs and failure causes. A generated source inventory is not a complete semantic code audit, a mock is not an autonomous Harness, and a smoke checkpoint is not a walking-policy benchmark.

Resource allocation is explicit and based on the experiment. The user permits single-node multi-GPU jobs and does not impose an artificial runtime or task-count cap. Never infer that reserving more GPUs automatically parallelizes a single-GPU trainer.
