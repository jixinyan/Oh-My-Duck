# Development conventions

## Boundaries and entry points

`omd.py` / `oh_my_duck.cli` is the public CLI. Project modules are
under `src/oh_my_duck/`, grouped by project capability; see [architecture.md](architecture.md).
`infrastructure/` owns setup, process dispatch and scheduler submission and imports
no simulator at startup. `rl/` owns task recipes, MDP, learners, export and evaluation;
`robotics/microduck/` owns models, assets, motor behavior and robot conventions.

Native acceptance belongs in `validation/{metric,harness,release}/`; terminal
Harness export belongs in `experience/harness_replay.py`. Use `omd validate --help`
and `omd replay --help` to find public entry points. Install the `validation` extra
for JSON Schema, saved image and ONNX audits. Actual physics checks additionally
require the locked simulator environment and the pinned native Harness dependency.
Reusable implementation imports package modules directly. Compatibility scripts
contain entry-point dispatch only; pytest does not add `scripts/` to its import path.

The native worker implementation lives in `integrations/edh/`: `environment.py`
owns observations and simulator stepping, `device.py` owns action execution,
`session.py` owns native session and tool boundaries, and `worker.py` owns process
transport. `integrations/edh_native.py` remains the executable and public import
entry point; see the [source map](../src/oh_my_duck/integrations/edh/README.md).

Asset preparation belongs in `rl/backends/isaac_newton/`; see its
[source map](../src/oh_my_duck/rl/backends/isaac_newton/README.md).
`assets.py --reuse-from-source` requires a clean complete ancestor checkout and
identical conversion inputs and resolved dependencies. It creates a new verified
artifact directory with original conversion status and explicit source provenance.
The existing source fingerprint and generated-file checks apply to every use.

Task factories and policy configuration are editable project source. Native RSL-RL
and SB3 PPO remain dependencies. Microduck task code is never loaded from the
upstream cache. `environments/{mujoco,isaac-newton,isaac-assets}` holds dependency
manifests and locks; run their setup through `omd setup`.

Newton setup includes the native Harness extra and the selected PPO framework.
The Linux x86_64 runtime uses `usd-exchange==3.0.0` (OpenUSD 26.08) as its unique `pxr` provider. Setup
reinstalls and verifies its files through `infrastructure/usd_runtime.py`, using
`.cache/tmp` for installer temporary files. `doctor --metadata-only` and online
backend construction perform the same file checks before CUDA initialization.
See [OpenUSD readiness](reports/openusd-readiness-2026-10-08.md).

USD instance expansion, source MJCF geometry mapping and physics material
preparation live in `task_binding/collision_assets.py`. This module imports only
the standard library until an actual asset is prepared. `task_binding/collisions.py`
owns simulator spawning and the Newton model-initialization callback.
`task_binding/collision_model.py` applies source contact masks, explicit ground
pairs and scene geometry corrections to the actual Newton builder. Its module
imports no simulator or learning library until called.
`omd validate model-assets --output DIRECTORY` runs actual USD preparation and
Newton imports with `CUDA_VISIBLE_DEVICES=''`. The output directory must be new;
the command retains partial results and per-model progress if validation fails.
Registered Newton training, evaluation and direct environment creation use
`verify_execution_runtime`. Native learner workers verify their own environment
before GPU selection or initialization; diagnostic entry points verify before
launching Isaac. MuJoCo paths retain their separate dependency environment.

The Isaac backend requires Newton. Unsupported task bindings fail explicitly.
Keep training dependencies out of agentic, voice, recording and CLI imports.

## Files and provenance

- Commit code, small reproducible configs, documentation, audit inventories and compact measured reports.
- Ignore upstream checkouts, Python environments, package caches, credentials, checkpoints, videos and raw trajectories. Obtain upstreams from the checked-in revision manifest.
- Each job runs from a detached Git worktree of its committed revision under `.job-sources/`; source imports and configuration point at that snapshot. Commit pending changes before submission. Environments, pristine upstream dependencies and output directories are shared through explicit links. Existing snapshots are never edited.
- Each job has a unique name and immutable output directory. Save source revision, dirty state, actual invocation, dependency versions, source/model hashes and results alongside outputs.
- Worker commands use shared absolute paths. Temporary files belong in the ignored project directory `.cache/tmp`; set `TMPDIR` explicitly.
- Large evidence stays under `outputs/` / `logs/`; link it from compact reports with hashes and job identifiers. A machine-local output path is not a portable published artifact.
- Keep upstream code and asset notices separate. The maintained Microduck source meshes are package data with preserved upstream ancestry. Generated USD and downloaded policy weights remain ignored.

## Version control

`main` tracks `origin/main` at `git@github.com:jixinyan/Oh-My-Duck.git`.
Remote fetch and push access are verified. Fetch and inspect remote history before
merging; never force-push or change the user's Git identity.

Use one focused active development branch: `feat/<topic>`, `fix/<topic>` or
`docs/<topic>`. At each completed, reviewable milestone, run relevant checks,
merge the branch into `main`, and push `main` before starting the next feature
branch from the updated `main`. Do not stack new work on an older feature branch
or leave completed fixes spread across branches. Merge verified infrastructure
independently of long-running training; document pending behavioral acceptance.

Check remaining branch tips for unmerged commits before cleanup. Delete local
branches only after their tips are contained in `main`; retain immutable training
worktrees, commits and experiment artifacts. Running training uses its pinned
source snapshot and does not follow the working branch.

Commit at coherent, reviewable checkpoints: interfaces/scaffolding, working behavior with its relevant checks, and measured validation evidence. Do not accumulate an entire feature into one final commit or make empty commits merely to increase frequency. Keep documentation describing a behavior in the same commit as that behavior. Record incomplete implementations explicitly rather than implying validation.

Before committing, inspect the staged diff and ensure no ignored artifacts or credentials entered the index. After committing, check `git status` and `git log`, and report the branch, commit IDs and whether they are local or pushed. Do not change the user's Git identity.


## Progress

Update the original design and execution documents when scope, module responsibilities or milestone status changes. Distinguish implemented, executed and validated. Record failed experiments, including job IDs and failure causes. A generated source inventory is not a complete semantic code audit, a mock is not an autonomous Harness, and a smoke checkpoint is not a walking-policy benchmark.

Resource allocation is explicit and based on the experiment. The user permits single-node multi-GPU jobs and does not impose an artificial runtime or task-count cap. Never infer that reserving more GPUs automatically parallelizes a single-GPU trainer.

**执行方式更新（2026-09-08，用户最新指示）**：单 GPU 开发、验证和训练直接在开发机 headless 运行。调度训练失败后，当前完整训练也获准使用本机空闲的 8 张 GPU；多卡 job 仍是可选执行方式。所有尝试使用独立输出目录和固定源码，保留此前失败记录。


## Diagrams

Use illustrative, editable SVG flowcharts in `docs/diagrams/` for new or revised
project diagrams. Embed them with relative Markdown image paths. The README's
[project overview](diagrams/project-overview.svg) and
[RL pipeline](diagrams/rl-pipeline.svg) establish the visual style: vector icons,
clear module groups, readable labels, directed connections and explicit external
or future boundaries. Keep diagrams focused on project scope; experiment progress
belongs in reports. Include SVG `<title>` / `<desc>` and descriptive Markdown alt
text. Use a self-contained viewBox and system font fallbacks; no scripts, remote
fonts or rasterized substitutes. Check XML, references and a rendered preview.

Task families keep `environment.py` and `ppo.py` together under `src/oh_my_duck/rl/tasks/<family>/`. See the [RL source map](../src/oh_my_duck/rl/README.md) and [full training campaigns](rl-campaigns.md).
