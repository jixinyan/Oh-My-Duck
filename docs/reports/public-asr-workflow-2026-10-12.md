# 公开 Qwen ASR 语料测量流程

固定源码 `0913bc5bc070f0d4ce50ef617201c16d418849b6` 提供三个公开入口：
`omd validate asr-prepare`、`omd validate asr-measure` 和
`omd validate asr-audit`。独立安装的项目包完成语料获取、全部实际转写和
原始音频复核，三个进程均正常退出。

## 完整录音与模型

使用 `google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd` 的完整
Mandarin Chinese 与 English test 语料，许可为 CC BY 4.0。
原始 Parquet 的文件尺寸、LFS SHA256 和全部录音数量均与固定来源一致。
945 条中文和 647 条英文录音全部参与测量，共 1,592 条、17,453.08 秒音频。

模型为 `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。
项目 `QwenSpeechRecognition.transcribe` 执行真实推理，全部模型参数为
CPU Float32；八个 CPU threads、一个 interop thread、batch size 1、
`max_new_tokens=512`，没有 language hint。CUDA visibility 为空，CUDA 未初始化。
Apple M5、24 GiB 内存、macOS 26.3 和 Python 3.12.12；模型加载耗时 28.784 秒。

| 语料 | 数量 | CER | WER | 转写耗时 | Real-time factor | 中位耗时 | P95 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mandarin Chinese | 945 | 7.681% | — | 1,580.544 s | 0.142839 | 1.501 s | 2.933 s |
| English | 647 | 2.323% | 5.306% | 994.377 s | 0.155666 | 1.435 s | 2.519 s |

全部 Float WAV 的采样值与原始录音一致。复核重新读取全部原始 Parquet、
录音身份、参考文字和实际 WAV，检查摘要、编号、完整数量与音频元数据，
然后重新计算全部错误率和耗时指标。独立复核通过，退出状态为零。
测量程序摘要为 `6660aadd2e400e564c32f1232b381df91a2f2728106dbc55ed837a9697eea023`。
完整逐条资料摘要为 `4bab52502490eb2b0e051a9f87833476c62effce281a072ecbf2069d8b71b640`。

## 安装与输入检查

公开安装说明使用 `environments/voice-asr/uv.lock` 和
`oh-my-duck[asr-evaluation]`。独立环境实际执行这项安装，模型与指标依赖
保持声明的版本；三个公开入口的帮助和完整语料准备均正常退出。
同一独立环境执行完整原始音频复核，五次公开命令均正常退出。

七项实际输入检查拒绝非空 CUDA visibility、零 CPU threads、未固定模型
revision、具有 tracked 修改的源码、重复语言、错误总数和错误文件摘要。
各项保留实际退出资料，输出目录未生成，模型推理未执行。
27 项模块导入与公开帮助检查通过。
Mac 独立安装通过 447 个源码与资源文件、三份许可证和 31 项公开命令。
Linux 独立安装核验 34 个实际依赖 wheel，依赖一致性检查通过；
447 个源码与资源文件、三份许可证和 31 项公开命令全部通过。
Linux 音频动态库实际加载，PortAudio 为 V19.6.0-devel，libsndfile 为 1.2.2。
Linux 安装检查没有执行 ASR 模型推理。

独立交接检查重新核验完整音频、全部指标、两个主机安装资料、实际输入拒绝
与所属进程身份。测量时记录的四个所属进程全部退出，所属 ASR 进程数量归零。

## 保存资料与使用

| 资料 | 路径 |
| --- | --- |
| 三个公开阶段、实际退出与完整复核 | `outputs/acceptance/public-asr-driver-20261012-01/result.json` |
| 原始语料准备 | `outputs/acceptance/public-asr-installed-preparation-20261012-01/result.json` |
| 全部录音、实际转写与指标 | `outputs/acceptance/public-asr-cpu-mac-20261012-01/` |
| 独立音频与指标复核 | `outputs/acceptance/public-asr-driver-20261012-01/independent-audit.json` |
| 公开安装命令 | `outputs/acceptance/public-asr-install-commands-20261012-01/result.json` |
| 七项输入拒绝 | `outputs/acceptance/public-asr-admission-20261012-01/result.json` |
| Mac 独立安装 | `outputs/acceptance/public-asr-mac-package-20261012-01/result.json` |
| Linux 独立安装与音频动态库 | `outputs/acceptance/public-asr-linux-package-20261012-04/result.json` |
| 完整流程、两个主机安装与进程退出 | `outputs/acceptance/public-asr-handoff-20261012-01/result.json` |

完整安装命令、参数和保存格式见 [ASR 使用说明](../asr-measurements.md)。
错误率衡量人声朗读录音的识别表现，机器人命令意图、噪声、机载麦克风和
完整交互延迟使用各自的实际样本与设备评估。GPU 验收与 RL 保持停止。
