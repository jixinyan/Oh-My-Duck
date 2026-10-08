# Isaac Lab and Newton backend

The backend uses the pinned Isaac Lab Newton implementation and official
Microduck timing, observations, named actuators and BAM behavior. Native simulator
libraries belong to the isolated `isaac-newton` environment; conversion uses
`isaac-assets`.

| Responsibility | Entry point |
| --- | --- |
| Runtime environment and simulator binding | `environment.py`, `physics.py`, `task_binding/simulation.py` |
| Official task binding and observations | `tasks.py`, `contracts.py` |
| Motor implementation | `bam_actuator.py` |
| Asset CLI and Kit conversion | `assets.py`, `convert_asset.py` |
| Source fingerprint and generated file verification | `paths.py` |
| Explicit reuse of identical converted assets | `asset_reuse.py` |
| Conversion reference and robot naming | `asset_reference.py`, `asset_names.py` |
| Source MJCF collision mapping and USD physics materials | `task_binding/collision_assets.py` |
| Newton collision masks, explicit ground pairs and scene geometry corrections | `task_binding/collision_model.py` |
| Official solver geometry materials and contact masks | `task_binding/contact_model.py` |
| Native MuJoCo Warp contact compilation and graph preparation | `task_binding/manager.py` |
| Simulator spawning and model-initialization callback | `task_binding/collisions.py` |

`collision_assets.py` compiles the official source model, expands USD instances,
maps each enabled collider to its source geometry and authors its physics material.
It imports no native simulator or USD module until called. `collisions.py` registers
the model-initialization callback and passes the active builder to
`collision_model.py`. Contact configuration can be applied directly to a real
CPU builder; two-world official model preparation and finalized arrays passed
the [CPU collision audit](../../../../../docs/reports/collision-model-2026-10-08.md).
`contact_model.py` configures the actual solver model; `manager.py` compiles its
native MuJoCo Warp contact tables before graph capture. Four CPU solver models
and their compiled data passed the
[contact model audit](../../../../../docs/reports/contact-model-2026-10-08.md).
The Linux kitless environment uses `usd-exchange==3.0.0`, which supplies OpenUSD
26.08. Setup and startup verify the unique provider and its installed files.

Run the public CPU asset audit from the locked Newton environment:

```bash
CUDA_VISIBLE_DEVICES='' omd validate model-assets --repeat 3 \
  --output outputs/acceptance/model-assets-unique-run
```

The command checks both robot variants by default. `--models` selects explicit
variants. It verifies actual geometry, physics materials, source joints, generated
file preservation and native Newton imports in anonymous USD session layers.
Results, dependency/source provenance and per-model progress are saved under the
new output directory. Solver execution and behavioral acceptance use the release
campaign.

`omd assets --backend isaac-newton -- --model MODEL` converts an asset when its
source fingerprint has no verified generated result. To reuse a converted asset
from an ancestor checkout with identical conversion inputs, provide that clean,
complete checkout explicitly:

```bash
CUDA_VISIBLE_DEVICES='' omd assets --backend isaac-newton -- \
  --model allcollisions --reuse-from-source /absolute/path/to/ancestor-checkout
```

Both checkouts must be clean committed source. Reuse checks all robot files,
conversion tools, bootstrap modules, upstream pins, conversion environment
configuration, resolved dependencies and every generated file. Only unactivated
project optional-dependency metadata is omitted from the dependency comparison.
Changed conversion inputs or resolved dependencies terminate the operation.
The destination must be new; existing artifacts are retained.

The copied build manifest preserves the original conversion and physical-validation
status. `source_reuse` records both Git revisions, fingerprints, input hashes,
dependency hash and original build hash. Reuse initializes neither Kit nor CUDA.
The standard `require_asset` source fingerprint and file checks remain required.
Actual Newton physics and policy behavior require the separate release campaign.
