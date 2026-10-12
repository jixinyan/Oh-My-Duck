# Qwen ASR 真实录音 CPU 测量

项目 ASR 实现使用固定源码 `7b94665bb116aeb411597f67c459b4f19647e901`，
通过 `QwenSpeechRecognition.transcribe` 转写全部录音。
模型为 `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。
十个模型文件的尺寸均与固定来源一致；权重文件的 SHA256 与上游 LFS 一致：
`79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`。

FLEURS 语料为 `google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd`，
许可为 CC BY 4.0。`cmn_hans_cn` test 的 945 条与 `en_us` test 的
647 条全部参与测量，共 1,592 条人声朗读录音、17,453.08 秒音频。
参考文字使用原始 `transcription`，同时保留 `raw_transcription`。

## 环境与推理

Apple M5、24 GiB 内存、macOS 26.3、Python 3.12.12。
模型参数全部位于 CPU，dtype 为 Float32；使用八个 CPU threads、一个
interop thread、batch size 1、`max_new_tokens=512`，没有 language hint。
CUDA visibility 为空，CUDA 保持未初始化。

锁定 ASR 环境使用 Torch 2.9.1、qwen-asr 0.0.6、Transformers 4.57.6、
Hugging Face Hub 0.36.2、NumPy 2.5.3、SoundFile 0.14.0 和 regex 2026.9.10。
语料与指标依赖为 PyArrow 21.0.0 和 Jiwer 4.0.0。
模型准备与加载耗时 130.040 秒，其中包含首次模型文件下载。
测量进程正常退出；记录的三个所属进程均已不存在。

## 全量结果

| 语料 | 录音数量 | CER | WER | 音频时长 | 转写耗时 | Real-time factor | 中位耗时 | P95 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mandarin Chinese | 945 | 7.681% | — | 11,065.18 s | 1,544.237 s | 0.139558 | 1.465 s | 2.849 s |
| English | 647 | 2.323% | 5.306% | 6,387.90 s | 980.938 s | 0.153562 | 1.433 s | 2.498 s |

文字规范化使用 Unicode NFKC、casefold、移除 Unicode 标点和合并空白。
CER 计算前移除空白，English 同时计算 WER。
错误率根据全部参考文字和实际转写汇总，保留 substitution、deletion、
insertion 和 hit 数量。Real-time factor 为转写耗时除以音频时长。
耗时按完整录音逐条测量。

## 独立复核

`omd validate asr-audit` 使用独立安装的 wheel，执行目录位于项目之外。
固定复核源码为 `7908436f6ed62f6f7a72be789bf180ed6f91265a`，实现位于
`src/oh_my_duck/validation/asr_corpus.py`。
复核从原始 Parquet 重新读取全部录音、参考文字和身份，检查文件 SHA256、
连续编号、完整数量、WAV 元数据与每个 Float32 采样值，再重新计算全部指标。
完整复核退出状态为零。

六项检查使用实际保存资料的独立副本，各自明确修改一项数据：

| 检查 | 结果 |
| --- | --- |
| 缺少最后一条转写 | 拒绝，退出状态 1 |
| 重复录音编号 | 拒绝，退出状态 1 |
| 参考文字与原始语料不一致 | 拒绝，退出状态 1 |
| 输入音频摘要不一致 | 拒绝，退出状态 1 |
| 规范化文字不一致 | 拒绝，退出状态 1 |
| 汇总错误率与逐条记录不一致 | 拒绝，退出状态 1 |

每项拒绝均保留实际错误、进程与退出资料，验收结果文件未生成。
25 项模块导入与 CLI 检查通过；模型依赖保持延迟加载。
独立安装检查通过 445 个源码与资源文件、三份许可证及 29 项公开命令。
用户的 Isaac asset lock 保持原有 SHA256。

## 资料与使用范围

| 资料 | 路径 |
| --- | --- |
| 固定语料、上游摘要与完整数量 | `outputs/acceptance/fleurs-corpus-preparation-mac-20261012-01/result.json` |
| 全部实际转写、音频与汇总 | `outputs/acceptance/fleurs-asr-cpu-mac-20261012-01/` |
| 完整复核、六项拒绝和进程退出 | `outputs/acceptance/fleurs-asr-cpu-independent-20261012-01/result.json` |
| 模型文件与权重来源 | `outputs/acceptance/fleurs-asr-model-source-20261012-01.json` |
| 测量环境 | `outputs/acceptance/fleurs-asr-cpu-mac-environment-20261012-01.json` |
| 测量进程退出 | `outputs/acceptance/fleurs-asr-cpu-process-exit-20261012-01.json` |
| 独立安装 | `outputs/acceptance/asr-corpus-audit-mac-package-20261012-03/result.json` |

公开复核参数与保存格式见 [ASR 语料复核](../asr-measurements.md)。
该结果测量人声朗读录音在项目 CPU Float32 路径上的识别表现，使用实际错误率
评估识别质量。机器人命令意图、噪声、机载麦克风、音质与完整交互延迟继续
使用相应的实际样本和设备验收。GPU 验收与 RL 保持停止。
