# Voice audio modules — 2026-10-08

Verified source: `05e337d9c1dd8bcc59de8db24b75a18bbfc596e0`.
Remote source: `.job-sources/voice-audio-20261008-01` on `jd_B300`.

## Source responsibilities

[`voice/audio.py`](../../src/oh_my_duck/voice/audio.py) owns local audio paths,
WAV decoding, finite-sample checks, SHA256 and exclusive PCM16 output.
[`voice/profiles.py`](../../src/oh_my_duck/voice/profiles.py) owns SQLite voice
revisions and active selection. [`voice/qwen.py`](../../src/oh_my_duck/voice/qwen.py)
owns model snapshots, device admission and serialized inference. HTTP services,
clients and audio devices import the shared audio operations directly.
The [voice source map](../../src/oh_my_duck/voice/README.md) identifies every module
and its public command.

All six shared audio functions retain their identifiers and complete computation.
Independent AST comparison against `7b48a16a64b5b59882cc68d3aaece122e353f1b6`
checked 24 functions and classes, including model adapters, HTTP services,
profile transactions and device behavior. Every compared definition matches.
The shared module SHA256 is
`130d956567287e9427b800f4c68a6419f01345f2f6cbfdae8da0628f53970568`.

## Profile and model checks

Ten actual WAV/SQLite tests passed using the official reference audio. They cover
candidate confirmation, persistent revision history, active selection, unchanged
confirmation requests, conflicting concurrent writes, changed or missing audio,
and nonfinite samples. Each locked Qwen environment passed 13 device-admission
tests, including 36 invalid constructor calls per environment.

Three actual model processes used the committed source, cached fixed snapshots,
`HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=''` and four PyTorch threads. All floating
parameters were CPU Float32. The model environments preserve their frozen locks.

| Model | Fixed revision | Parameter count | Actual execution |
| --- | --- | --- | --- |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` | 914643008 | Two utterances with the confirmed profile |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `5ecdb67327fd37bb2e042aab12ff7391903235d3` | 1916676352 | One candidate from the requested description |
| `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 782426112 | Reference audio and all three generated WAV files |

The active profile remains `语音验收小鸭` / `已确认音色一号` / revision 1,
with reference SHA256
`e70207c688d248a1b51feddf3661586abdeb830d255d68d46e49353ff24f3861`.
Full profile contents and SQLite database bytes remained unchanged after each
model phase. VoiceDesign generated a candidate without changing active selection.

## Audio verification

An independent process decoded the original reference and three generated WAV
files through SoundFile. All four files have complete PCM16 frames, 24 kHz mono
samples, finite nonzero signals and matching original SHA256. The ASR transcripts
match all four requested texts after ignoring punctuation and whitespace.

| Audio | Duration / frames | ASR text |
| --- | --- | --- |
| Confirmed reference | 2.96 s / 71040 | 你好，我是小鸭。今天我们一起玩吧。 |
| Confirmed-profile speech | 3.20 s / 76800 | 我会保留这个声音，下一次继续陪你说话。 |
| Confirmed-profile feedback | 2.16 s / 51840 | 任务已经完成，我停下来了。 |
| New candidate | 3.12 s / 74880 | 你好，我是小鸭。今天我们一起玩吧。 |

These four utterances verify the executed inference and audio path. Recognition
accuracy, perceived voice quality and interactive latency require their own
evaluation. The [CPU voice navigation report](cpu-voice-navigation-2026-10-08.md)
records actual native task execution and its agentic MP4.

## Independent installation and resource release

The wheel and source distribution passed installation outside the checkout:
434 required source/resource files, three licenses and 26 public CLI checks.
Installed audits also recomputed the retained 1208 official ONNX actions and
verified the saved five-motion physical sequence.

The three inference processes, PIDs 428631, 434478 and 435769, exited with code 0.
Both admission processes, PIDs 437959 and 438320, exited with code 0. The
independent check found none of these PIDs active. CUDA remained uninitialized;
GPU acceptance and RL remain stopped. Robot, RL, native physical execution,
model settings and voice-profile transaction behavior retain their preceding
verified implementations.

## Retained artifacts

Paths below are relative to the repository root. Actual generated WAV files,
model logs, constructor-test logs and process receipts remain in their matching
directories.

| Evidence | Path |
| --- | --- |
| 24 original definitions and shared-module identity | `outputs/acceptance/voice-audio-structure-20261008-01/result.json` |
| Ten WAV/SQLite tests | `outputs/acceptance/voice-audio-structure-20261008-01/profiles-test.log` |
| Three actual model phases | `outputs/acceptance/voice-audio-models-20261008-01/result.json` |
| Two locked admission environments | `outputs/acceptance/voice-audio-admission-20261008-01/result.json` |
| Independent audio, profile, source and process audit | `outputs/acceptance/voice-audio-independent-20261008-01/result.json` |
| Built distributions and installed commands | `outputs/acceptance/voice-audio-20261008-01-install/result.json` |

Model-result SHA256:
`393d903db01234f1ac1ace7ea427685659f279145fcd983628bb0dc7b22be84e`.
Admission-result SHA256:
`317203a6a2fbe50bbede168ad5ee0432394630c134c8effbe359be032e9ae44d`.
Wheel SHA256:
`dae3115ae062ffa62cc46910cb74ee1519d96311bedb351d7dbe8611faf512eb`.
Source-distribution SHA256:
`2b5a4c69d78d31cf75636a1a2a010c5d7c879d9c64cfb9ed4ad623884ce16e89`.
