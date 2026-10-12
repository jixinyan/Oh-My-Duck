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

`voice-task --audio FILE FILE --context-previous-task` 在同一原生 session
依次执行录音指令，明确引用上一项成功任务。交互式 `execute_recording`
接受 `context_run_ids`。来源和归属检查、当前物理状态、持久经验读取
与正式验证由原生 Harness 管理。

Public entry points are `omd voice`, `omd voice-task` and `omd voice-session` in
[`cli/`](../cli/). See [voice interaction](../../../docs/voice-interaction.md),
[confirmed profiles](../../../docs/voice-profiles.md) and the
[architecture source map](../../../docs/architecture.md#source-organization).

FLEURS 中文与英文全部 1,592 条真实录音及识别指标已通过独立复核，见
[语料测量](../../../docs/reports/asr-corpus-2026-10-12.md)。
公开复核入口为 `omd validate asr-audit`，位于
[`validation/asr_corpus.py`](../validation/asr_corpus.py)。它读取原始 Parquet、
全部转写记录与实际 WAV，重新计算识别错误率及耗时指标。
语料准备与测量入口为 `omd validate asr-prepare` 和 `omd validate asr-measure`，
分别位于 `validation/asr_preparation.py` 与 `validation/asr_measurement.py`。
