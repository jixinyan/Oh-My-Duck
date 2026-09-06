# Owned Microduck source migration — 2026-09-06

The user requires Microduck tasks, rewards, actor/critic configuration and related implementations to be editable parts of Oh My Duck. The previous cached-upstream delegation has been replaced for Microduck code. Source is in `training/microduck/src/omd_microduck`, with Apache-2.0 license, original guidance and file-level provenance from official commit `29e887ecfbf5d37144759e5a9f8a176dfb83d547`.

## Completed in this change

- Imported task recipes, MDP implementations, model assets, BAM extensions, runner, normalized export, manifest and CPU rehearsal as maintained source. Preserved baseline task semantics and removed the upstream HF auto-submission hook.
- Extracted the RSL runner to `rl/rsl_runner.py`; introduced editable native SB3 policy class/kwargs in `rl/sb3.py`. RSL actor/critic settings remain editable in each local recipe.
- Switched both MuJoCo framework environments to the maintained package, uninstalling cached `mjlab-microduck`. Isaac now consumes the same local BAM and robot source. Native simulator/PPO libraries remain dependencies; Isaac Lab's generic launcher remains external.
- Updated both environment locks without changing physics or PPO pins. The unused `rustypot` package was removed from MuJoCo environments. All three training environments installed the local package successfully.
- Switched MuJoCo export and CPU replay to maintained modules. Publishing provenance identifies `jixinyan/Oh-My-Duck`; original ancestry is recorded separately. No upload or W&B sync occurred.
- Documented customization and future task registration in `training/microduck/README.md`; synchronized the original design/execution documents and maturity status.

## Checks after migration

- All 33 compiled task inventory entries match the pre-migration official inventory after normalizing only the Python namespace. This compares catalog fields, not every runtime trajectory or all config parameters.
- Lightweight framework/interface tests: **17 passed**.
- Maintained upstream MDP/manifest and native SB3 tests: **53 passed**.
- Installed Isaac configuration/terminal-boundary tests: **7 passed**.
- Explicit import audit: maintained runner/MDP/rehearsal resolve inside this repository; `mjlab_microduck` is absent and no loaded module comes from cached `microduck_rl`.

The first combined pytest invocation failed because lightweight tests intentionally assert simulator modules are not imported; run them in a separate process. It also found the old asset fixture path and original publisher console-script expectation, both corrected. The initial RSL-environment pytest invocation lacked pytest; the locked test dependency group was installed in the SB3 environment used for regression checks. Failed attempts are not counted as passing checks.

## Earlier evidence, before this source switch

Both representative tasks passed native RSL-RL dual-GPU smoke (64 environments per rank, five iterations), with offline W&B and separate rank seeds. The following jobs completed successfully while the old implementation was active:

| Job | Evidence |
|---|---|
| `omd-walk-ddp-export-0906-01` | Reward audit and normalized runner export: `outputs/walk-ddp-export-20260906-01/` |
| `omd-stand-ddp-export-0906-01` | Reward audit and normalized runner export: `outputs/standup-ddp-export-20260906-01/` |
| `omd-standup-sb3-resume-0906-01` | 7,680 → 15,360 timesteps; 768 timeouts captured; reward audit passed; reload error 4.77e-6 |
| `omd-standup-sb3-export-0906-02` | Normalized ONNX parity maximum error 9.54e-7 |

Native dual-GPU launcher logs reported successful workers but included an incomplete trailing rank-1 traceback header with no exception after completion. Check this cleanup/logging anomaly during the next native resume run. It has not been diagnosed. These short checkpoints do not demonstrate learned walking or recovery.

## Resume here

1. Re-run 64-environment / five-iteration MuJoCo smoke from the maintained source for Walking and StandUp with both native frameworks, audit rewards, then check native continuation and normalized export. Prior GPU evidence does not automatically validate the new import route or checkpoint compatibility.
2. Rebuild Isaac walk and ground-contact assets: changing source paths changes the build fingerprint. Old outputs are preserved and deliberately not treated as current cache hits. Re-run BAM/contact checks against the maintained source.
3. Implement the full Isaac Walking/StandUp task binding, including actual contacts/sensors, observation delays, commands, rewards, DR, curricula, terminal observations and reset behavior. Only the Isaac PD diagnostic is currently registered for training; local source migration does not change that limit.
4. Consolidate task discovery and per-backend capability registration. Current task registration is local and editable, but the CLI catalog still selects IDs containing `MicroDuck`. Splitting the inherited monolithic MDP module into focused modules is follow-up work, with parity tests.
5. Measure native PPO throughput at larger environment counts, then train and evaluate useful Walking/StandUp policies. Add task-specific behavior and sim2sim batteries/videos; no learned-behavior acceptance is claimed yet.

The user requested stopping at “usage 55.” The tools expose no account usage/cost meter; its unit and current value were requested. No new long training or full Isaac migration was started after that instruction. This checkpoint supports stopping and resuming without losing the source migration or overstating verification.
