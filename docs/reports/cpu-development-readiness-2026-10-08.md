# CPU development readiness

The CPU work covers native policy execution, measured motion, cancellation,
transport, recorded perception, voice profiles, asset preparation, dependency
integrity, release inputs and independent installation. Each report identifies
the actual source revision and retained artifacts. GPU acceptance and RL remain
stopped.

| Capability | Verified scope | Evidence |
| --- | --- | --- |
| Native Luna office task | Actual OpenAI model, official policy and sensors; passed formal Verifier, 569 controls, 2276 substeps, 1040 events, 113 observer frames, 12 progress checks, full MP4 decode and resource release | [CPU model navigation](cpu-luna-navigation-2026-10-08.md) |
| Native execution retries and goal evidence | Three executions preserve complete physical state; 200 controls, 800 substeps, six rejected starts and three read-only checks; independent ONNX parity, 18 import tests, 433 installed files and 26 CLI calls | [Retry validation](native-execution-retry-2026-10-08.md) |
| Metric tool parameters | Native units, speeds and turn translation; actual Luna four-motion execution, 17 import tests, 432 installed files and 25 CLI calls | [Tool guidance](metric-tool-guidance-2026-10-08.md) |
| Native metric request lifecycle | Eight rejected calls preserve the prepared request and physical state; four caller deadlines retain execution; two motions and two explicit replacements pass 641 controls, 2564 substeps, independent ONNX checks and resource release; 432 installed files and 25 CLI calls | [Metric admission](metric-request-admission-2026-10-08.md) |
| Native task wait deadlines | Finite positive API and voice CLI admission, 19 actual checks, HTTP resource closure and independent installation | [Wait admission](native-wait-admission-2026-10-08.md) |
| Configured perception sources | Native tool schema, environment metadata and scene tool agree; 42 Python/Node cases, 48 tests, two actual CPU sessions, 150 controls and 600 substeps; installed package verified | [Source selection](perception-source-capabilities-2026-10-08.md) |
| Current CPU scene perception | Five actual software-rendered states, 1253 native geometry intersections, target distances/bearing, unchanged physical state during reads; 431 installed files and 24 CLI calls | [CPU perception](cpu-scene-perception-2026-10-08.md) |
| Native training inputs | Integer counts, finite search fractions, actual CLI admission and unchanged committed campaigns; 110 Linux tests, three subtests, 430 installed files and 24 CLI calls | [Campaign admission](campaign-input-admission-2026-10-08.md) |
| CPU voice control and interruption | Actual Qwen ASR → OpenAI Luna → native policy/sensor tools → confirmed-voice interruption feedback; 1400 controls, 5600 substeps, 2427 events, 296 images, two decoded WAV files, unchanged profile and complete resource release | [CPU voice task](cpu-voice-task-2026-10-08.md) |
| Qwen CPU inference | Three fixed models, 26 admission tests, 72 rejected constructor calls, six installed unavailable-CUDA calls, three decoded WAV files, confirmed-profile preservation and process exit; ASR text matches 2/3 generated utterances | [CPU voice validation](qwen-cpu-validation-2026-10-08.md) |
| Application and tool interfaces | Actual native CPU session, task catalogue, resource release, seven complete JSON Schemas, 77 tests, 33 subtests and 430 installed files | [Native application interfaces](native-application-interfaces-2026-10-08.md) |
| Configurable native scenes | Shared CPU/Newton schema; actual Python CLI and pinned Node startup/exit, seven declared configurations, 23 rejected variants, 64 tests and 431 installed files | [Scene configuration](native-scene-configuration-2026-10-08.md) |
| Native robot worker | Actual CPU MuJoCo/BAM motion, pinned SDK/process transport, ActionGate, policy transitions, cancellation and resource release | [Worker validation](native-worker-modules-2026-10-08.md) |
| Training-package tools | Actual retained Walking/StandUp packages, configured locomotion selection, complete episodic duration and transition; 725 controls, 145 frames, 87 stopped samples; 429 installed files and 24 CLI calls | [Package integration](policy-packages-2026-10-08.md) |
| Native head/body commands | Actual software-rendered CPU execution; 575 controls, positive/negative head pitch and body height response, zero-command return, 115 camera frames, 95 stopped samples and installed independent audit | [Pose validation](native-pose-2026-10-08.md) |
| Motion and policy evidence | Continuous five-motion sequence; 1208 independently recomputed ONNX actions with zero error; physical states and stopping checked | [Validation modules](validation-modules-2026-10-08.md) |
| Sensor and voice records | Actual saved RGBD validation, frame identity, distance reconstruction and confirmed voice profiles | [Offline validation](offline-release-validation-2026-10-07.md) |
| Converted assets | Both variants, complete conversion inputs and dependencies, source fingerprints and every generated file | [Source-verified assets](source-verified-assets-2026-10-08.md) |
| OpenUSD and native imports | Unique patched OpenUSD 26.08 provider; six imports across three processes; source geometry, materials and joints preserved | [OpenUSD validation](openusd-readiness-2026-10-08.md) |
| Newton contact configuration | Actual two-world source masks, explicit ground pairs and finalized CPU arrays for both robot variants; 27 tests and 422 installed files | [Collision model validation](collision-model-2026-10-08.md) |
| Official solver contact data | Four actual Newton CPU solver constructions and MuJoCo Warp compilations; ten contact fields, twelve native arrays and 21 protected fields checked per model; 28 tests and 423 installed files | [Contact model validation](contact-model-2026-10-08.md) |
| External scene geometry | Actual Office and Hospital imports, all 5682 geometric colliders, 164 signed-scale transformations and two finalized CPU models | [External scene validation](external-scene-cpu-2026-10-08.md) |
| Public CPU asset command | Four actual Newton imports, passive wheel joints, ordered progress, failed-result retention and output protection | [Public asset validation](model-assets-cli-2026-10-08.md) |
| Newton execution entry points | Nine actual dependency failures before CUDA/Isaac initialization; actual locked setup repair; 26 tests and 421 independent installed files | [Execution preflight](newton-execution-preflight-2026-10-08.md) |
| Current-source release preparation | Six stages, four scenes, ten policies, 29 pinned Harness files and actual model-provider metadata; CUDA uninitialized | [Execution preflight](newton-execution-preflight-2026-10-08.md) |
| Installed execution lifecycle | Actual completion, explicit deadlines, subprocess interruption, retained command records and existing-output protection | [Installed lifecycle](installed-audit-lifecycle-2026-10-08.md) |

Source responsibilities are listed in the [architecture source map](../architecture.md#source-organization).
`integrations/edh/` owns native environment, device, session and transport;
`validation/` owns acceptance; `experience/` owns replay export;
`task_binding/collision_assets.py` owns USD preparation;
`task_binding/collision_model.py` owns Newton contact configuration;
`task_binding/contact_model.py` owns official solver material and contact masks;
`robotics/policies/` owns common joint ONNX admission and the training-package catalogue;
`infrastructure/usd_runtime.py` owns installed OpenUSD verification.
Public execution and audit commands remain available from installed packages.
All verification processes exited. Source, configuration, wheel, source
distribution and evidence hashes are recorded in the corresponding reports.

## GPU acceptance scope

Actual Newton solver execution, initialization with freshly compiled kernels,
rendering, current-controller multi-scene motion, ordered navigation, model
perception accuracy and learned-policy behavior require their own execution
evidence. The [release campaign](../runtime-release-acceptance.md) declares the
six sequential runtime stages and preserves source, models, physical state,
formal task verdicts, media and resource release.

Future authorized execution permits at most one physical GPU from devices 2–4.
The selected device must have zero compute PIDs and sustained zero utilization.
Existing workloads and other users' processes are preserved. Hardware acceptance
requires an actual Microduck.
