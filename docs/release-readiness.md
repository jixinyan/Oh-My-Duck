# Release readiness

The release workflow uses recorded speech, the pinned native Harness, official policies, actual simulation sensors, independent task verification, and the confirmed voice profile. Each acceptance run has a unique directory and preserves source revisions, model identity, task events and measured physics.

## Runtime preparation

Use Python 3.12, Node.js and the locked environments in `environments/`. Install the voice client on the operator computer and the ASR, TTS and Isaac/Newton environments on the NVIDIA host. Keep the model services bound to localhost and use SSH forwarding between hosts. Voice setup requires an explicitly confirmed profile before task feedback can be synthesized.

```bash
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice-client --locked --no-editable
environments/voice-client/.venv/bin/python omd.py doctor --runtime voice-client \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --harness-url http://127.0.0.1:4318 --output outputs/acceptance/doctor-client-new.json
```

The doctor checks installed dependency versions and the actual service identities. On the simulation host, `--runtime isaac-newton --scene-config configs/simulation-demo/office.json --catalog POLICY_DIRECTORY` also checks CUDA, the pinned Isaac Lab revision, the converted robot asset and the complete official ONNX catalogue. Readiness checks and behavioral acceptance have separate evidence.

## End-to-end acceptance

```bash
environments/voice-client/.venv/bin/python omd.py voice-task \
  --audio /absolute/path/command.wav --persona duck \
  --harness-url http://127.0.0.1:4318 \
  --profile nvidia-office-6.0 --scenario navigate-nvidia-office-6.0 \
  --output outputs/acceptance/voice-task-new
```

`voice-task` preserves the recorded audio SHA256 and ASR model revision, submits the recognized instruction unchanged with the native catalogue digest, waits for the actual task result, and synthesizes a status response using the active voice profile. It returns exit code 0 for a successful native task and 2 for a completed task that failed acceptance. Service and transport errors terminate execution. The command produces audio files; interactive device playback is provided by `voice-session`.

Acceptance requires a real native model task, actual pretrained policy actions, current sensor observations, a passed independent Verifier result, confirmed execution termination and session cleanup. Metric motion checks also require measured distance/angle error and five upright stopped samples. Operator interruption must confirm the native execution boundary and demonstrate that control and physics counters remain unchanged afterwards. Physical braking and execution-clock interruption are recorded separately.

## Evidence available on 2026-10-03

| Capability | Evidence | Scope |
|---|---|---|
| Recorded speech → native agent → Newton → Verifier → fixed voice | Run `5dfb18d9-b31b-4226-bc94-9391ddd847ac`, succeeded; target error 0.090916 m; 480 control steps | NVIDIA Office, one seed, simulator ground truth |
| Standard-foot metric tools | 0.5 m request: 0.024196 m final error; 45° request: 0.522914° error; five stopped samples | Official `alpha_walking`, Newton/BAM and native ActionGate |
| Runtime readiness | Local voice client and remote Newton doctor passed | Installed dependencies, actual model services, scene assets and ten-policy catalogue |
| Agentic video with speech | 91.4 s, 1920 × 1080 MP4, full decoding passed | Actual public events and camera frames; recorded input and confirmed-voice feedback |

The [acceptance report](reports/end-to-end-2026-10-03.md) records paths, revisions and hashes. RL remains stopped. Hardware behavior, recognition accuracy, object carrying, image-only VLN, RTX rendering and broad scene generalization require their own acceptance. Release scope and public capability statements must use these evidence boundaries.
