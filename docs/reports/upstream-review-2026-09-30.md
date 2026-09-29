# Official Microduck upstream review — 2026-09-30

The official repositories were fetched from their public remotes on 2026-09-30:

| Repository | Project pin before review | Current upstream head | Decision |
| --- | --- | --- | --- |
| [microduck_rl](https://github.com/pollen-robotics/microduck_rl) `develop` | `29e887e` | `cfe1c2a` | Pin updated; safe reset fix migrated |
| [microduck](https://github.com/pollen-robotics/microduck) `main` | `bc41fb5` | `f0d934e` | Pin updated; runtime contract already represented by owned Python gates |

The RL diff is mostly a September protective-fall / VelStand development line. Its useful generic simulator fix is that all absolute reset heights must be offset by each vectorized terrain origin. The old owned event layer wrote local heights directly into world `qpos`, which is harmless on a plane and wrong on raised rough terrain. `set_random_ground_state`, `set_random_crouch_state`, prone resets, and roulade resets now apply the origin offset through `_env_origin_z`; a CPU regression test covers a nonzero origin and a selected-env reset. The owned reset API also now carries the upstream side-fall and post-fall servo-joint randomization hooks, disabled by default for the representative recipes.

The upstream protective-fall line also adds side-fall and post-fall joint randomization, servo impact/stall costs, expert behavior cloning, warm-start curriculum handling, and an all-collision robot variant. Those changes are task-specific and alter training semantics. They remain available as a separately reviewable future VelStand/protective-fall variant; they are not silently applied to the representative Flat Walking or Flat StandUp baselines. The all-collision backlash asset and servo-geometry naming work already present in the working tree remain preserved for that variant.

The latest runtime repository adds defensive ONNX loading, explicit recurrent-state support, and factory-servo provisioning checks. This framework's policy path is Python, so the relevant load-time guarantees are maintained in `rl.artifacts.publish.manifest`: one graph input/output, fixed 61→14 widths, finite-action smoke execution, schema-2 metadata, and normalizer-aware export provenance. Copying the Rust runtime would duplicate the external daemon rather than improve the training framework.

`configs/upstream.json` and this file record the reviewed heads. `third_party/microduck_rl/UPSTREAM.json` keeps the owned-source ancestry and records the selective migration. The official upstream formula and task semantics remain the baseline; policy experiments must still pass the full smoke/export/rehearsal/unforced gates before a policy can become a tool.
