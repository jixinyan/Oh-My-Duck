# Development conventions

## Boundaries and entry points

`omd.py` / `oh_my_duck.cli` is the public CLI. `src/oh_my_duck/` holds the complete application framework and interfaces; see [architecture.md](architecture.md). `scripts/` contains acquisition, process dispatch and job submission; it must remain usable without importing a simulator. Backend implementation belongs under `training/<backend>/`. `training/common/` holds only interfaces and data semantics shared by both simulators, never implicit simulator defaults.

The MuJoCo training algorithm remains upstream: the launcher delegates to its installed trainer and exporter. Local headless evaluation imports the pinned official inference implementation. Changes to source pins, policy format, physics or command semantics require corresponding design and evaluation updates.

The Isaac backend must use Newton. A fallback to PhysX is a scope change, not a silent compatibility fix. Keep `.envs/mujoco` and the future `.envs/isaac-newton` independent.

## Files and provenance

- Commit code, small reproducible configs, documentation, audit inventories and compact measured reports.
- Ignore upstream checkouts, Python environments, package caches, credentials, checkpoints, videos and raw trajectories. Obtain upstreams from the checked-in revision manifest.
- Each job has a unique name and immutable output directory. Save source revision, dirty state, actual invocation, dependency versions, source/model hashes and results alongside outputs.
- Worker commands use shared absolute paths. `/tmp` is for disposable research, not distributed-job dependencies.
- Large evidence stays under `outputs/` / `logs/`; link it from compact reports with hashes and job identifiers. A machine-local output path is not a portable published artifact.
- Keep upstream code and asset notices separate. Do not copy meshes or weights into the Git repository merely to make an example self-contained.

## Version control

Local `main` is initialized with `origin=https://github.com/jixinyan/Oh-My-Duck.git`. Remote inspection failed with the current credentials; no remote history is assumed and no force-push is allowed. When access becomes available, inspect/fetch remote history and reconcile it before pushing local commits.

Use focused commits for scaffolding, behavior changes and validation evidence. Before committing, inspect staged files and ensure no ignored artifacts or credentials entered the index. Do not change the user's Git identity.

## Progress

Update the original design and execution documents when scope, module responsibilities or milestone status changes. Distinguish implemented, executed and validated. Record failed experiments, including job IDs and failure causes. A generated source inventory is not a complete semantic code audit, a mock is not an autonomous Harness, and a smoke checkpoint is not a walking-policy benchmark.

Resource allocation is explicit and based on the experiment. The user permits single-node multi-GPU jobs and does not impose an artificial runtime or task-count cap. Never infer that reserving more GPUs automatically parallelizes a single-GPU trainer.
