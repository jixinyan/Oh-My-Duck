# Configured perception sources

Native perception tools advertise the sources supported by the current scene.
Every simulator exposes `simulator_ground_truth`. A validated
`perception_endpoint` enables `models` in the tool parameter schema, environment
metadata and `microduck.scene_info`. Requests for an unavailable source fail
before camera acquisition. Model service errors propagate through the native
tool response.

The source revision is `c824a89129cbaf5b8b4c87c363c42c9ca9e0170c`.
Python and the pinned Node deployment agreed on 42 actual configuration cases:
ten accepted and 32 rejected. The focused scene suite passed 48 tests, including
both simulation backends and invalid endpoint types, addresses, ports and paths.

Two actual CPU MuJoCo/BAM native sessions used the official apartment and pinned
`velstand` policy. Each executed 75 admitted controls and 300 physical substeps,
then reached a confirmed paused boundary. Both environment metadata and the
scene tool returned their exact configured sources. Six unavailable-source
requests were rejected. Two requests to an actual closed local port returned
connection errors. Both sessions subsequently read actual visible targets through
the explicitly requested simulator source. Joint state, controls, simulation time,
sequence and stopped-sample counts remained unchanged during those reads.
Native policy and control servers closed; the acceptance process exited with
code zero. CUDA remained uninitialized.

The independent installed wheel and source distribution passed verification of
431 source/resource files, three licenses and 24 public CLI calls outside the
checkout. These checks verify source selection and native tool behavior;
navigation success, model recognition, Newton execution and hardware retain their
own acceptance requirements. GPU acceptance and RL remain stopped.

Evidence is retained in:

- `outputs/acceptance/perception-source-admission-20261008-02/result.json`
- `outputs/acceptance/perception-source-capabilities-20261008-02.json`
- `outputs/acceptance/perception-source-capabilities-20261008-02-install/result.json`

Reproduce the native CPU check with `scripts/accept_harness_perception.py`, using
the pinned physical Harness runtime, official policy directory and a new output
path. The script runs real policies and native control requests.
