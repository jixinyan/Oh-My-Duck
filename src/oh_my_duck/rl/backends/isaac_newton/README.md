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
