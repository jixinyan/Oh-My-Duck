# Qwen 文件语音推理验收

2026-09-26 在 `jd_B300` 的 GPU 5 上运行。两套环境均使用各自的 `uv.lock` 安装，`torch` 与 `torchaudio` 使用 CUDA 13 的 2.9.1 版本。ASR 环境包含 `qwen-asr 0.0.6`、`transformers 4.57.6`；TTS 环境包含 `qwen-tts 0.1.1`、`transformers 4.57.3`。源代码在服务器的 `.cache/voice-source` 独立副本中验证，模型缓存与语音资料位于被 Git 忽略的 `.cache` 目录。

官方 `asr_zh.wav` 的 SHA256 为 `46dbc998c9d1d48111267c40741dd3200f2e5bcf4075f8c4c97f4451160dce50`。`Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0` 在 GPU 上转写为“甚至出现交易几乎停滞的情况。”，与官方示例文本一致。

`VoiceDesign` 使用“温暖、清楚、轻快的中文声音”和“你好，我是小鸭。今天我们一起玩吧。”生成 2.96 秒候选音频。用户确认动作执行前，资料库没有活动音色。确认后，`duck-001` 的活动版本为 1，参考音频 SHA256 为 `e70207c688d248a1b51feddf3661586abdeb830d255d68d46e49353ff24f3861`。两个独立合成进程均读取这一版本及同一参考音频。活动版本与参考音频 SHA256 在两次合成后保持一致。

| 操作 | 模型 commit | 加载耗时 | 推理耗时 | 音频长度 | CUDA 峰值预留显存 |
|---|---|---:|---:|---:|---:|
| VoiceDesign 候选 | `5ecdb67327fd37bb2e042aab12ff7391903235d3` | 285.55 秒，含首次下载 | 29.22 秒 | 2.96 秒 | 4,334,813,184 字节 |
| Base 合成“你好，我们开始今天的冒险吧。” | `5d83992436eae1d760afd27aff78a71d676296fc` | 13.69 秒，权重已预先下载 | 49.15 秒 | 2.48 秒 | 2,447,376,384 字节 |
| Base 合成“我会记住这个声音，下一次还用它和你说话。” | 同上 | 10.84 秒 | 38.62 秒 | 3.52 秒 | 2,499,805,184 字节 |
| ASR 转写首段合成音频 | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 12.95 秒 | 21.80 秒 | 2.48 秒 | 1,895,825,408 字节 |
| ASR 转写第二段合成音频 | 同上 | 12.49 秒 | 40.23 秒 | 3.52 秒 | 1,897,922,560 字节 |

首段合成音频的回读文本与输入完全相同。第二段回读为“我会记住这个声音。下一次还用它和你说话。”，文字相同，句中停顿对应的标点有所变化。生成音频均为非空 WAV，采样值有限，SHA256 与保存结果一致。试听文件保存在 `outputs/voice-validation-20260926/`：`reference.wav`、`greeting.wav`、`persistent-voice.wav`。

### 延迟观察

Base 模型位于 `cuda:0`，attention implementation 为 `sdpa`。同一个已加载模型、同一文本、`torch.get_num_threads() == 1` 时，连续两次合成分别耗时 8.86 秒与 4.24 秒；模型加载耗时 13.39 秒。另一组运行中，默认 128 个 CPU 线程的首次合成耗时 25.07 秒，随后改为 1 个线程的同文本合成耗时 2.99 秒。模型预热和线程数量同时变化，这组数据仅说明线程争用值得控制。另一次单次 CLI 调用在 1 个线程设置下耗时 82.65 秒，当前数据不能保证固定推理延迟。独立语音 CLI 默认使用 1 个 CPU 线程，适配器不会修改调用者进程的线程设置。

### 检查

本地使用官方 Qwen3-TTS 参考 WAV 运行 `PYTHONPATH=src .cache/voice-profiles/test-venv/bin/python -m pytest tests/test_voice_profiles.py -q`，结果为 `10 passed`。候选音色在确认前没有活动版本，重启资料库后仍可读取；确认后活动版本和参考音频 SHA256 可再次读取。语音文件的数值检查由存储和输出路径执行；合成内容通过独立 ASR 回读核查。

`uv lock --check` 对 `environments/mujoco`、`environments/isaac-assets`、`environments/voice` 和 `environments/voice-asr` 在本地通过。`environments/isaac-newton` 在含完整 IsaacLab 缓存的远端隔离源码副本中通过相同检查。`environments/isaac-assets` 的实际暂存版本只包含 voice extra 元数据；该版本另在远端隔离副本通过 `uv lock --check`，其他未提交的 marker 变化未计入验收。本地缺少 IsaacLab 源码目录，无法直接完成 Newton 锁的本地检查。`git diff --check` 通过。

文件转写、声音设计、显式确认、版本持久化和文件合成已在目标 GPU 验证。麦克风采集、扬声器播放、播音中断和外部 Harness 接入尚未实现。`async` 接口使用工作线程运行模型；取消等待不会终止正在执行的 GPU 推理。
