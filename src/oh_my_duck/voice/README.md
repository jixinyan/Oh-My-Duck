# Voice source map

| Responsibility | Source |
| --- | --- |
| Voice profiles, candidates and audio-device interfaces | `base.py` |
| Local audio paths, WAV decoding, finite samples, SHA256 and PCM16 output | `audio.py` |
| Confirmed voices, candidate persistence, revision history and active selection | `profiles.py` |
| Qwen ASR, VoiceDesign, Base TTS and inference-device admission | `qwen.py` |
| ASR and TTS HTTP services | `service.py` |
| HTTP client, transcription identity and synthesized-audio metadata | `remote.py` |
| Microphone recording, speaker playback and device interruption | `device.py` |
| Interactive requests, native task submission, interruption and episode events | `session.py` |

`audio.py` contains the shared file operations used by models, profile storage,
HTTP services and audio devices. It reads WAV through SoundFile, validates every
decoded sample and hashes the original bytes. Model output is saved as PCM16 WAV
with a local file URI and SHA256.

`profiles.py` owns SQLite transactions, confirmed revisions and the content-addressed
reference audio directory. `qwen.py` owns model snapshots, model identity, CPU/CUDA
admission and serialized inference. ASR and TTS run in their separately locked
environments. The native Harness owns task planning and tool selection.

Public entry points are `omd voice`, `omd voice-task` and `omd voice-session` in
[`cli/`](../cli/). See [voice interaction](../../../docs/voice-interaction.md),
[confirmed profiles](../../../docs/voice-profiles.md) and the
[architecture source map](../../../docs/architecture.md#source-organization).
