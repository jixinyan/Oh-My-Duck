# Native CPU pose validation

`omd validate pose` runs the official `alpha_stand` policy in the furnished
CPU MuJoCo/BAM apartment through the pinned native Harness, its control tools
and ActionGate. `omd validate pose-audit` independently checks the saved records.
Implementation belongs to `validation/harness/pose.py` and `pose_records.py`.

## Execution requirements

Use the locked CPU apartment environment and the pinned native Harness source.
The native Python package must be available through its source directory or
installed in the execution environment. `--edh-source` identifies the checkout
containing the native physical schema. Use software rendering on Linux and set
`CUDA_VISIBLE_DEVICES` to an empty value. The campaign checks the actual OpenGL
renderer for Mesa `llvmpipe` or `softpipe`. The OSMesa shared library and its
dependencies must be installed or provided through `LD_LIBRARY_PATH`.

Run from the project checkout with a new output directory:

```bash
export TMPDIR="$PWD/.cache/tmp"
export CUDA_VISIBLE_DEVICES=''
export MUJOCO_GL=osmesa
export PYTHONPATH="$PWD/src:$EDH_SOURCE/harness/physical-runtime/src"
python -m oh_my_duck validate pose \
  --scene-config configs/simulation-demo/apartment-metric.json \
  --catalog "$OFFICIAL_POLICY_DIR" --edh-source "$EDH_SOURCE" \
  --output outputs/acceptance/native-pose
```

The campaign requires clean committed source and preserves the source revision,
runner hash, scene hash, official manifest hash and actual runtime versions.
Existing output directories are protected.

## Physical schedule and evidence

The native worker executes 75 warmup controls with `velstand`, selects
`alpha_stand` at the confirmed paused boundary and runs five phases:

| Phase | Head pitch command | Body height command | Controls |
| --- | --- | --- | --- |
| Neutral | 0 rad | 0 m | 100 |
| Positive | +0.2 rad | +0.01 m | 100 |
| Return positive | 0 rad | 0 m | 100 |
| Negative | −0.2 rad | −0.01 m | 100 |
| Return negative | 0 rad | 0 m | 100 |

All other command slots remain zero. Each phase records actual named servo
positions, body position, head RGB and observer RGB. Every action retains its
official ONNX input, command, output and matching native control identity.
The campaign finishes through `finish_policy` and confirms session cleanup.

The audit requires all 575 controls and 2300 physics substeps, exact command
admission, 61/14/13 dimensions, official ONNX parity, upright endpoints, zero
external obstacle contacts and five measured stopped samples. Saved observer
PNG bytes and timestamps must match their original native events. Head RGB
must decode at the declared dimensions.

Physical response uses the final 25 controls of each phase. Positive and
negative head pitch changes must each exceed 0.03 rad in the requested
direction; body height changes must each exceed 0.001 m. Returning the command
to zero must produce a smaller error relative to that phase's preceding neutral
measurement. These criteria establish measurable response and return for the
two commanded quantities. Other head/body axes, precise tracking, Newton
execution and hardware each require separate physical evidence.

```bash
python -m oh_my_duck validate pose-audit \
  --directory outputs/acceptance/native-pose \
  --catalog "$OFFICIAL_POLICY_DIR" \
  --output outputs/acceptance/native-pose-audit.json
```
