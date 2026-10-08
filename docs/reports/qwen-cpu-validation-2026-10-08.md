# Qwen CPU inference — 2026-10-08

Verified source: `c6387504045a702507bc39051da1bca50640b1d5`.
Remote source: `/home/jixin/workspace/code/Oh-My-Duck/.job-sources/qwen-cpu-20261008-01`.
The ASR and TTS environments were installed separately from their frozen locks,
including the declared test group. Model snapshots were read from the existing
Hugging Face cache with `HF_HUB_OFFLINE=1`. Every inference used
`CUDA_VISIBLE_DEVICES=''`, CPU Float32 parameters and four PyTorch threads.

## Device admission

All three Qwen adapters validate `device` before importing their model SDK or
requesting a snapshot. The CPU route retains Float32; the CUDA route retains
BFloat16. Unsupported device values terminate at admission.

Both locked environments passed 13 actual tests, including 36 invalid
constructor calls each. Six additional calls against the installed wheel
rejected unavailable CUDA before either model SDK was imported. No output audio
directory or CUDA context was created by those rejected calls.

## Actual model inference

| Model | Fixed revision | Executed result |
| --- | --- | --- |
| `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 2.96 s reference audio; exact stored text; 4.600935 s inference |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` | Two utterances using the same confirmed profile; 2.56 s and 3.36 s WAV; 11.443515 s and 12.546216 s inference |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `5ecdb67327fd37bb2e042aab12ff7391903235d3` | 2.88 s candidate WAV; 11.033347 s inference |

ASR parameter count was 782426112; Base was 914643008; VoiceDesign was
1916676352. Every recorded floating parameter was `torch.float32` on `cpu`.
The environment versions were `torch==2.9.1+cu130`, `qwen-asr==0.0.6`,
`qwen-tts==0.1.1`, `soundfile==0.14.0` and `huggingface-hub==0.36.2`.
ASR used `transformers==4.57.6`; TTS used `transformers==4.57.3`.

The active profile remained `语音验收小鸭` / `已确认音色一号` / revision 1,
with reference SHA256
`e70207c688d248a1b51feddf3661586abdeb830d255d68d46e49353ff24f3861`.
Its full profile and SQLite database bytes were unchanged after every model run
and the independent ASR process. VoiceDesign produced a candidate file. The
existing active profile remained selected.

## Audio and text evidence

Independent decoding checked all three generated files: WAV PCM16, 24 kHz,
mono, complete frame counts, finite samples, nonzero signal and exact SHA256.

| Generated text | Duration / frames | ASR text |
| --- | --- | --- |
| 你好，今天我们继续冒险吧。 | 2.56 s / 61440 | 您好，今天我们继续冒险吧。 |
| 我会保留这个声音，下一次继续陪你说话。 | 3.36 s / 80640 | 我会保留这个声音，下一次继续陪你说话。 |
| 你好，我是小鸭。今天我们一起玩吧。 | 2.88 s / 69120 | 你好，我是小鸭。今天我们一起玩吧。 |

Two of three texts match after ignoring punctuation and whitespace. The first
record retains the `你好` / `您好` difference. The evidence establishes actual
inference, audio encoding and confirmed-profile reuse. Recognition accuracy,
perceived voice quality and interactive latency require separate evaluation.

## Installed package and resource release

The wheel and source distribution passed independent installation outside the
checkout: 430 source/resource files, three licenses and 24 public CLI calls.
VoiceDesign, the three-audio ASR process, the active-profile CLI and the
unavailable-CUDA checks used installed modules. The installed Qwen module matches
the verified source SHA256
`402c7924f86e8a2423c64df30a7e6e329d0fb860c284fe6515a51b94358c6d52`.
All 438 wheel member contents, including distribution metadata, match between
the local and remote builds.

The four inference processes, PIDs 366017, 374060, 399076 and 406964, exited.
The subsequent process and compute-process checks found none of those PIDs.
CUDA remained uninitialized. GPU acceptance and RL remain stopped.
The only production-source change from `d54040d7` is `voice/qwen.py`;
robot, RL, native physical execution and voice-profile persistence preserve
their preceding implementations.

## Retained artifacts

Paths below are relative to the repository root. Raw audio and inference logs
remain under the matching `outputs/acceptance/qwen-cpu-*` directories.

| Artifact | SHA256 |
| --- | --- |
| `outputs/acceptance/qwen-cpu-admission-20261008-02/result.json` | `36b4e7c18622e5239b23ee3b39912d891b1ab2adddc5300e2ebe26a6b3415e4b` |
| `outputs/acceptance/qwen-cpu-asr-20261008-01/result.json` | `f22dd687b59627bdae5014033dfefa3f8ae96bbbf58766ce36f6ec1ee6ef39ad` |
| `outputs/acceptance/qwen-cpu-tts-20261008-01/result.json` | `b4272a14c1c35fb1b340c3e1afb2afce3c08d7c6b0c8126c9381b0489b1d141c` |
| `outputs/acceptance/qwen-cpu-design-20261008-01/result.json` | `e63f5d6b9ec394abb191baf5e75ede9ee16445fe61245b3eccc853fe133cdb23` |
| `outputs/acceptance/qwen-cpu-roundtrip-20261008-01/result.json` | `5625cbbc59fca0977f62024fcdee2f0fe1e1bce15f030ab6408ef9d46b1d7175` |
| `outputs/acceptance/qwen-cpu-installed-admission-20261008-01.json` | `1c352069eaef350320409728033acb851871ebb668e133f703cea9176b62bc7b` |
| `outputs/acceptance/qwen-cpu-install-20261008-01/result.json` | `ca3e494ce5bfc7a4facdb424f8f8305a67a84e2b844b02f27b858930c84896b0` |
| `outputs/acceptance/qwen-cpu-processes-20261008-01.json` | `1a417e00302e6ea11cb3dc41da996614b67e96d6d12cd0d7f6963d1ad0febcdc` |
| `outputs/acceptance/qwen-cpu-independent-20261008-01.json` | `af217cfb2dce7e55857db91d91fdcd23665b1417ee60082f738c9f9de6a7663c` |

Local wheel SHA256:
`8978039afbf3fe1354cff5b23da1f19dc23c7e23a830ddd5d8628305b69a1ee9`.
Remote wheel SHA256:
`e71cb9d71c46a1ca4f24bcfbf5f7377764e52a4c96ff2d1532842ee9d4424002`.
Source distribution SHA256:
`8e2219bdbb0101c0c4be03670725307e33ba450804cd4f045c98fb4b8f98de03`.

## Reproduction

Install the two locked environments and run their actual admission tests:

```sh
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice-asr --frozen --group test
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice --frozen --group test
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/.cache/tmp" environments/voice-asr/.venv/bin/python \
  -m pytest tests/voice/test_model_admission.py -q
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/.cache/tmp" environments/voice/.venv/bin/python \
  -m pytest tests/voice/test_model_admission.py -q
```

For CPU file inference, use the corresponding environment and specify
`--device cpu --cpu-threads 4` in the [voice commands](../voice-profiles.md).
`transcribe --model-revision` fixes the ASR commit; `synthesize` reads the
confirmed Base revision from the active profile. Retain `HF_HOME` and
`HF_HUB_OFFLINE=1` when using prepared snapshots without network access.
The [CPU readiness record](cpu-development-readiness-2026-10-08.md) and
[runtime campaign](../runtime-release-acceptance.md) define remaining execution
acceptance.
