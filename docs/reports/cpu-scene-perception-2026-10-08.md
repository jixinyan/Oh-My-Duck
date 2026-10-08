# CPU apartment perception

Runtime source: `25111e55157f0f01efc9bdd4cb11c6cffb7d980d`.
Installed-package source: `d8e16728e1840eafc0aad1266375724f52a73dd7`.
All production files are identical between these pins.

The CPU MuJoCo/BAM backend implements `perception_frame` and `inspect_scene` for
the same native Harness tools used by Newton. `perception/mujoco.py` owns render
geometry and apartment labels. The backend retains physical ownership and frame
identity. Both `microduck.inspect_scene` and `microduck.observe` can request
`source="simulator_ground_truth"`; `source="models"` sends the calibrated frame
to the explicitly configured perception service.

The frame contains actual head RGB, per-pixel world points, current camera/body
positions, yaw, episode and sequence. World points use MuJoCo optical-axis depth
and the actual mono render frustum. Background and clipped pixels carry invalid
depth. Named apartment geometry and actual segmentation determine visible
targets. Compound furniture shares one target ID; results include the current
visible bounds, surface-distance median and interval, horizontal distance and
bearing. Unrepresented or invisible targets return an empty list.

## Actual software-rendered checks

Five states were executed in the official apartment: corridor, office, kitchen,
living room and corridor after 75 official `velstand` control actions. All 14 BAM
servos, 61 observations and 50 Hz control conventions remain intact. The last
case additionally required measured stopping before perception.

The process used `llvmpipe (LLVM 15.0.7, 256 bits)` and left CUDA uninitialized.
Independent pixel directions use the model camera's FOV and world transform;
MuJoCo native geometry intersection checks the surfaces selected by actual
segmentation, including the render near plane. 1253 pixels passed the 2 mm
comparison. The maximum measured error was 0.0018864104602376753 m.

Target labels, bounds, visible-pixel counts, valid depth, median distances, world
positions and bearing passed independent reconstruction. Each frame retained
episode and sequence. Reading perception preserved qpos, qvel, control values,
simulation time, action count, stopping samples and goal-hold samples. Subsequent
RGB was byte-identical, confirming render modes returned to RGB. Unknown targets
returned empty results. Renderer resources closed and the process exited with
code 0.

Eight invalid prompts and two foreign-thread calls were rejected before creating
a renderer or advancing physics. All 41 named target groups bind to geometry in
the actual compiled apartment model; these admission checks also closed their
resources and left CUDA uninitialized.

The local independent audit verified all 15 saved frame/inspection/geometry
artifacts, reconstructed all 1253 comparisons and checked source equality.
Independent wheel/source-distribution installation outside the checkout verified
431 source/resource files, three licenses and 24 public CLI calls.

| Artifact | SHA256 |
| --- | --- |
| `outputs/acceptance/cpu-scene-perception-20261008-03/result.json` | `9ff04f5551b897af88149e58e1b6c2a01fb5f28ce1b9d6b8b481fc56384e3a6a` |
| `outputs/acceptance/cpu-scene-perception-20261008-02-install/result.json` | `f66f03a0a8f3f46ca55d661e0f35fb8e6bfcc1a6c41b4664009db28229e56c2c` |

This verifies CPU render geometry and native backend interfaces. Actual model
recognition, complete agent navigation, Newton execution and hardware retain
their own acceptance requirements. GPU acceptance and RL remain stopped.

## Reproduction

Use the prepared CPU environment with OSMesa and llvmpipe libraries:

```sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  GALLIUM_DRIVER=llvmpipe TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src \
  python scripts/accept_cpu_perception.py \
  --catalog .cache/official-policies-runtime/1b56c396825c052a4e26e95cf2b8d8298af9e9b4 \
  --output outputs/acceptance/cpu-scene-perception-new
```

The output directory must be new. Records preserve source revision, original
RGB, annotations, world points, segmentation, native intersections and hashes.
Tool use and available labels are documented in the
[perception guide](../perception-navigation.md).
