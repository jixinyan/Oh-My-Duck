# Server migration handoff — 2026-09-21

GitHub `jixinyan/Oh-My-Duck` is the source remote. At the initial audit, local
main and origin/main both point to `55ec67f`; the working tree and stash are
empty, and all existing remote feature branches are merged into main. Migration
handoff updates are committed and pushed separately. No RL work is restarted.

## Data that Git push does not carry

The migration set includes the entire project tree, not only tracked files:

- `outputs/`, `logs/`, `wandb/`: native RSL checkpoints, complete SB3
  model/VecNormalize/metadata bundles, exports, videos, offline W&B data,
  experiment manifests, failed attempts and scheduler records.
- `artifacts/`: converted robot/scene resources and provenance.
- `.job-sources/`: immutable training snapshots and their local state.
- `.envs/` and `.cache/`: all five installed environments, managed Python,
  pinned reference checkouts and dependency caches, preserved as recovery aids.
- `.git/`, tracked source, documentation, all three `uv.lock` files, and other
  ignored project files. An independent Git bundle also preserves repository
  refs and detached worktree commits.

Installed environments and dependency cache share many hardlinks. Their combined
unique size is approximately 45 GiB; counting separate `du` invocations can
mislead. Other data is approximately 18 GiB. Archive size is recorded in the
backup receipt, not inferred from these estimates. Backups preserve hardlinks
and symlinks without dereferencing worktree links back into the repository.

## Backup package and verification

Prepared outside the repository at:
`/mnt/data/users/heyang/workspace/migration-backups/oh-my-duck-20260921/`.
This directory is private (0700); offline logs and local metadata should not be
uploaded to GitHub or synchronized to the saved W&B account.

- `project-state.tar.zst`: project tree except `.envs` and `.cache`.
- `environments-cache.tar.zst`: `.envs` and `.cache` together, preserving shared
  hardlinks; these are optional for a clean rebuild but retained to avoid loss.
- `repository.bundle`: final Git refs plus source-snapshot commits, independently
  verified with Git. Use this or GitHub for the latest handoff revision.
- `*-installed.json`, `git-*.txt`, `host.json`: actual installed versions, refs,
  worktrees and migration context.
- `SHA256SUMS` and `verification.json`: archive/bundle integrity and verification
  outcome. The presence of an archive alone is not proof of a completed backup.

Archive creation, compressed-stream validation, and archive-to-source comparison
must finish before the package is marked verified. This copy is on the same
`/mnt/data` filesystem as the project: it protects the handoff/package structure,
**not loss of that filesystem**. Copy the complete package to retained storage
or the destination server and verify it there before deleting the old copy.
Destination transfer is not claimed until a target is supplied and checked.

## Restore on the new server

1. Copy the entire backup directory. Run `sha256sum -c SHA256SUMS` inside the
   copied directory. Keep the original until all checks pass at the destination.
2. Extract into a **new empty directory**, not over an existing project:
   `tar --zstd -xf project-state.tar.zst -C /new/empty/project` and, if retaining
   installed dependencies, `tar --zstd -xf environments-cache.tar.zst -C /new/empty/project`.
   Never use `tar -h`: worktree links would duplicate shared outputs/environments.
3. Retrieve latest source from GitHub or `git clone repository.bundle /new/source`.
   The bundle is a second independent source recovery route; GitHub/bundle may
   contain a later verification receipt than the project-state archive's commit.
   Preserve the extracted original tree while reconciling source revisions.
4. Prefer restoring the old canonical runtime path
   `/mnt/data/users/heyang/workspace/code/oh-my-duck`. Existing run metadata,
   worktree `.git` pointers, Python executable links, editable package installs
   and some generated assets use absolute paths. `/home/heyang/workspace/code/oh-my-duck`
   is also an observed alias on the old host. Inspect symlinks before use.
5. If the root path changes, rebuild virtual environments from the committed
   `environments/*/uv.lock` through `python omd.py setup --help`, preserving pins
   and optional framework extras. Consult exported installed-version inventories
   for the original isolated official-baseline environment and overrides.
   Recreate worktrees with `git worktree repair`/the source-snapshot machinery
   only after reviewing their paths. Do not mass-rewrite preserved historical
   provenance JSON; document old-to-new path mapping separately.
6. Reinstall/verify GPU drivers and headless rendering dependencies, and arrange
   the new scheduler integration. Old `/usr/local/bin/submit` delegates to an
   administrator-managed `/root/alaya-htrain/submit`; that service is not part of
   this project backup. Home SSH keys, platform credentials and unrelated
   projects are outside this project-scoped migration.
7. Run CPU/source checks first. Before new full training, use a complete job with
   required gates, automatic continuation and final evaluation. Keep Isaac on
   Newton, native PPO semantics and **W&B OFFLINE**. Do not automatically resume
   suspended or regressing learners merely because the server changed.

## RL state to preserve

On September 14, `omd-walk-pacing-0913-01` became Suspended. On September 21 the
scheduler reports its task no longer exists in the API and marks its local entry
Deleted. This is not training completion. Local campaign JSON can still say
`running`; it is stale. Checkpoints and evidence remain local.

Walking progress/checkpoint inventory and policy assessment:
[paused checkpoint handoff](reports/rl-walking-suspension-2026-09-14.md) and
[policy review](reports/rl-walking-policy-assessment-2026-09-14.md).
All eight current Walking combinations remain below acceptance; complete budgets
and final unforced/transfer/CPU-BAM evaluation were not reached. Preserve both
original controls and experimental schedules. The older Newton RSL StandUp final
policy remains the strongest existing result (native seed-42 four-pose passes;
14/17 full CPU/BAM pose batteries), with prone robustness unresolved.
