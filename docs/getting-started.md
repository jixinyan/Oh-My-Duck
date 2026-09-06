# Getting started

These are the current development commands. For the whole product, start with the root README and Project Design. For actual results, see [implementation-status.md](implementation-status.md).

## Prepare the official backend

```bash
python omd.py --help
python omd.py setup
```

Setup pins upstream source revisions and downloads the official walking model. It creates Python 3.12 at `.envs/mujoco` without changing the system interpreter. The first download is several GB. Isaac/Newton has separate training and asset-conversion environments; see [Isaac integration](isaac-newton.md) for its diagnostic-only implementation and remaining migration work.

## Run server jobs

On this server GPU work must use Alaya HTrain:

```bash
python omd.py submit --name omd-probe-001 --gpus 1 -- \
  python omd.py probe --backend mujoco --output outputs/probe-001

python omd.py submit --name omd-smoke-001 --gpus 1 -- \
  python omd.py train --backend mujoco -- \
  Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 \
  --agent.max-iterations 5 --agent.logger tensorboard

python omd.py submit --name omd-eval-001 --gpus 1 -- \
  python omd.py eval --backend mujoco --output outputs/eval-001 --video
```

Use unique job names and output directories. Add `--dry-run` before the command separator to inspect a submission. Use `submit --status <job>`, `submit --logs <job>` and `squeue` to inspect work.

The wrapper requests one node and an explicit GPU count (1, 2, 4 or 8). Multi-GPU training additionally needs the backend's distributed flags; the official trainer accepts `--gpu-ids all`. Per-process environment counts affect total parallelism. No artificial task-count or runtime cap is added.

On a personal GPU machine, run `python omd.py train ...` / `eval ...` directly. Keep GPU work inside jobs on this shared server.

## Evaluation artifacts

The shared `configs/eval/flat_walk.json` protocol defines 14 seconds of hold → forward → stop → turn left → stop at 50 Hz. Commands use simulation ticks rather than wall-clock timing.

| File | Contents |
|---|---|
| `config.json` | Actual commands, timing, initial state, BAM parameters and fall definition |
| `result.json` | Completion/failure, velocity RMSE, contact-foot speed and provenance |
| `trajectory.npz` | Actual inference inputs, actions, targets, joint state and contacts |
| `rollout.mp4` | Optional EGL offscreen video |

A fall returns code 2 and preserves the failure report. The current CPU rehearsal runs the walking policy alone, without recovery, extra action filtering or added delay. Its collision model is the official CPU scene, not an assertion of training-scene identity.

A five-iteration checkpoint only validates training plumbing. Official pretrained weights are used for the initial motion baseline. Numeric ONNX consistency and cross-simulator task performance are separate checks.

## Entry points and modules

```text
omd.py                         public command entry point
src/oh_my_duck/infrastructure/bootstrap.py           fixed sources/models and isolated installation
src/oh_my_duck/infrastructure/run.py                 process dispatch into backend environments
src/oh_my_duck/infrastructure/submit.py              scheduler submission and provenance
src/oh_my_duck/robotics/microduck/protocol.py    shared timing and joint contract
src/oh_my_duck/rl/backends/mujoco/probe.py        worker GPU / EGL check
src/oh_my_duck/rl/evaluation/mujoco.py         official ONNX + BAM headless replay
configs/                       source pins and evaluation protocols
tests/                         meaningful contract and behavior checks
docs/                          design, execution, audit and measured results
```

Large files stay under ignored `.cache/`, `.envs/`, `artifacts/`, `outputs/` and `logs/`. Backend modules and future voice/tools packages are added when implemented, not as empty capability claims.
