# Server migration handoff — 2026-09-21

## Scope: source and reproducible configuration

The user's final migration decision is **code only**. Push owned source, robot
assets tracked in Git, task/experiment configurations, environment manifests and
locks, upstream pins/licenses, tests, diagrams and development reports to
[`jixinyan/Oh-My-Duck`](https://github.com/jixinyan/Oh-My-Duck).
Checkpoints, normalizers, videos, training logs, offline W&B runs, installed
virtual environments and dependency caches are outside the requested transfer.
They remain on the old filesystem; Git cannot restore these excluded artifacts.
Re-running training produces new runs, not the original checkpoint provenance.

At the audit, `main` and `origin/main` were both `55ec67f`, with no uncommitted
source or stash. All existing remote feature branches and all 27 detached job
source commits are ancestors of main. Migration documentation is committed and
pushed afterward. The original full-data archive operation was stopped after
the scope clarification; its generated archives were removed. Original training
artifacts were not deleted or uploaded. No training was restarted.

## Restore the project

```bash
git clone https://github.com/jixinyan/Oh-My-Duck.git
cd Oh-My-Duck
python omd.py --help
python omd.py setup --help
```

Use the documented setup entry point for the required backend and native PPO
framework. The three committed locks under `environments/`, root `pyproject.toml`
and `configs/upstream.json` retain dependency/pin inputs. Cached upstreams and
installed environments are recreated by setup; owned Microduck tasks/assets stay
under `src/oh_my_duck/rl` and `src/oh_my_duck/robotics/microduck`.

GPU drivers, headless rendering support and scheduler integration must be
available on the new server. The old `/usr/local/bin/submit` delegates to an
administrator-managed `/root/alaya-htrain/submit`; that service and home
credentials are not project source. Keep Isaac on Newton and **W&B OFFLINE**.
Inspect any absolute paths in historical reports before using example commands;
old run paths describe evidence on the original server, not portable artifacts.

A small local `repository.bundle` and its SHA-256 checksum are an optional second
Git recovery route under
`/mnt/data/users/heyang/workspace/migration-backups/oh-my-duck-20260921/`.
Only a successfully verified bundle is usable. GitHub main is the primary
migration destination; the local bundle is on the old data filesystem.

## RL state to preserve in source documentation

On September 14, `omd-walk-pacing-0913-01` became Suspended. On September 21 the
scheduler reports its task no longer exists in the API and marks its local entry
Deleted. This is not training completion. Local campaign JSON can still say
`running`; it is stale. Do not automatically resume it after migration.

The [paused checkpoint handoff](reports/rl-walking-suspension-2026-09-14.md) and
[policy review](reports/rl-walking-policy-assessment-2026-09-14.md) retain the
observed progress, recipes and debugging conclusions. Their checkpoint/video
links reference excluded old-server artifacts and will not resolve in a fresh
clone unless separately retained.

All eight current Walking runs remain below acceptance; complete budgets and
final unforced/transfer/CPU-BAM evaluation were not reached. Preserve both
original controls and experimental schedules in configuration. The older Newton
RSL StandUp final policy was the strongest observed result (native seed-42
four-pose passes; 14/17 full CPU/BAM pose batteries), with prone robustness
unresolved. This records prior evidence, not a claim that a fresh training run
will reproduce the same result or that representative RL acceptance is complete.
