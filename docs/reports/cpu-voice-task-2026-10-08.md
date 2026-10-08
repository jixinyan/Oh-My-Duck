# Recorded CPU voice control with Luna

The installed voice client completed recorded Qwen speech → actual ASR → the
pinned native Harness → OpenAI `gpt-6-luna` with `high` reasoning → official
policy and sensor tools → native interruption → confirmed-voice Qwen feedback.
The actual native Responses model check returned text and a normal completion.
OpenAI's model inventory exposed the Luna identifier `gpt-6-luna`.

## Native execution

Source `d8e16728e1840eafc0aad1266375724f52a73dd7` ran CPU MuJoCo/BAM in an
immutable Linux worker checkout. The native Harness revision was
`8a5e685b22d032207f53db20454f0992a4ad60fd`; the official policy catalogue was
`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`.
The actual apartment retained 14 BAM XL330 M6 servos, 61 actor observations and
four physical substeps per admitted control. GPU visibility was empty.

Qwen ASR submitted this transcription unchanged:

> 请使用距离和角度工具前往办公室。读取相机和距离传感器，到达后停止。

Native run `a9225b39-025b-477e-9c56-0436736d4bf5` executed 1400 policy calls and
1400 admitted controls, with 5600 physical substeps. It used actual policy
selection, distance/angle commands, camera observations, simulator-ground-truth
perception and sensor-based recovery. Navigation remained near the corridor
entry and the operator ended the diagnostic through the native stop API.
The terminal state is `cancelled`, with a device-confirmed ended execution and
`user_stop`. There are zero formal Verifier results. Office navigation remains
pending.

The export retains 2427 events and 296 original PNG files. Independent checks
verified every image's SHA256, byte count, dimensions and successful decoding,
and reconstructed the complete 0–1400 control count with exactly four substeps
per control. Configured-source admission is verified separately in the
[perception-source report](perception-source-capabilities-2026-10-08.md).

## Fixed voice and resource release

ASR used `Qwen/Qwen3-ASR-0.6B` at
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. TTS used
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at
`5d83992436eae1d760afd27aff78a71d676296fc`. Both ran CPU Float32 in separate
frozen environments. Input speech and interruption feedback used the same
confirmed voice ID and profile revision. The reference audio and profile database
bytes remained unchanged.

Independent decoding verified two complete mono PCM16 WAV files at 24 kHz:
144000 instruction frames and 28800 feedback frames, with finite nonzero samples
and matching SHA256. This check used recorded audio, without microphone capture
or speaker playback. Seventy-two actual health observations checked both services
during the diagnostic.

Native session `99d28d12-51a8-4156-a919-3763036f0b2a` reports `closed` with
`resources: released`. All five owned processes exited: the native server and
remote ASR/TTS services returned zero; the two SSH transports returned their
explicit cancellation code 130. Cleanup took 0.020–2.969 seconds, with no SIGKILL
or cleanup timeout. Local ports 55385–55387 and remote ports 19861–19862 are
closed. The matching remote native worker is absent. GPU acceptance and RL remain
stopped.

## Retained evidence

Artifacts are under `outputs/acceptance/cpu-voice-loop-20261008-03/`:

- `model-check.json`: actual native OpenAI Responses completion.
- `task/result.json`: transcription, native terminal record and fixed-voice feedback.
- `replay/`: complete events, original camera files and terminal export.
- `independent-lifecycle-audit.json`: execution counts, image and WAV validation,
  voice preservation, service health and process release.
- `remote-cleanup.json`: remote process absence and closed service ports.

The provider inventory is retained separately in
`outputs/acceptance/cpu-luna-provider-models-20261008-01.json`.
Actual voice control and interruption lifecycle are verified. Navigation success,
model recognition accuracy, current-source Newton execution and hardware require
their own acceptance evidence.
