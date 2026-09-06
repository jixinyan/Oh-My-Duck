# Headless replay validation — 2026-09-06

The architecture is merged locally into `main`; follow-up work is on
`feat/rl-pipeline-validation`. All eight representative training/resume/export/
local-package combinations have passed their lifecycle checks. Replay is an
independent acceptance gate, and short checkpoints are not learned policies.

## Pixel readback investigation

MuJoCo EGL on this shared H200 host can leave the output buffer untouched. The
native Python renderer allocates it with `np.empty`, so nonuniform garbage can
look like a successfully rendered image to a range-only assertion. A minimal
sphere scene reproduced the problem with a buffer initialized to 17. Selecting
only NVIDIA's EGL vendor produced two successful isolated runs, but subsequent
runs failed; neither vendor selection nor ONNX thread settings is a sufficient
fix. The precise driver cause remains unresolved. Earlier MuJoCo EGL videos
are not accepted as visual evidence; their independently recorded physics metrics
remain available.

`rl/evaluation/rendering.py` initializes the readback buffer and rejects empty
frames. `mujoco_video.py` renders the exact compiled model and qpos/qvel in a
separate process. The renderer is explicitly selectable (`egl` or `osmesa`);
there is no silent renderer or physics fallback. Isaac retains native Newton
video. MuJoCo uses native OSMesa software rasterization for acceptance on this
host. This changes image generation only, not simulation, rewards, BAM, PPO or
policy inference.

The optional Ubuntu 22.04 amd64 library is installed under `.cache/render-libs`,
without modifying system packages. Its URL/version/SHA256 are recorded in
`configs/rendering.json`; extraction includes the package's copyright notice.
Required system libraries are checked with `ldd`. Other systems can provide
compatible system OSMesa instead. No upstream checkout or installed Python
package is patched.

## Evidence

- `outputs/rehearsal-osmesa-0906-01`: CPU/BAM Walking battery completed with valid
  1280×720 video; the policy fell, so the result is `behavior_failed`.
- `outputs/eval-osmesa-0906-01`: native MuJoCo task replay completed with valid
  video; result `behavior_failed`.
- The CPU batch passed 27 lightweight tests and 66 task/SB3 tests, including a
  regression that rejects a renderer which never writes the output buffer.
- Full eight-policy replay/sim2sim validation follows this source checkpoint.

Failed attempts remain in their original directories, including the earlier
EGL, thread-setting and isolated-EGL-worker experiments. The initial SB3 matrix's
last CPU rehearsal was interrupted after the readback defect was confirmed;
its training/resume/export/package results were retained.
