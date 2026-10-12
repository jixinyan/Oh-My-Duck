# ASR 语料准备、测量与复核

`omd validate asr-prepare` 获取固定 FLEURS test 语料，核验上游 LFS SHA256、
文件尺寸、许可与完整录音数量。`omd validate asr-measure` 使用项目的
Qwen ASR 在 CPU Float32 执行全部转写，保存逐条录音、来源、文本和耗时。
`omd validate asr-audit` 独立复核保存资料。

模型运行使用 `environments/voice-asr/uv.lock`；语料与指标依赖由
`oh-my-duck[asr-evaluation]` 提供。将这项 extra 安装到对应的 ASR 环境。
测量使用固定且没有 tracked 修改的 Git 源码目录，安装的 Qwen 与测量模块
必须和该目录的文件摘要一致。

```bash
omd validate asr-prepare \
  --cache-directory .cache/fleurs-corpus \
  --dataset-revision 70bb2e84b976b7e960aa89f1c648e09c59f894dd \
  --languages cmn_hans_cn en_us \
  --output outputs/asr-corpus-NEW

CUDA_VISIBLE_DEVICES='' omd validate asr-measure \
  --corpus outputs/asr-corpus-NEW/result.json \
  --source-root /absolute/path/fixed-project-source \
  --model-revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --cpu-threads 8 \
  --output outputs/asr-measurement-NEW

omd validate asr-audit \
  --capture outputs/asr-measurement-NEW \
  --corpus outputs/asr-corpus-NEW/result.json \
  --source-root /absolute/path/fixed-project-source \
  --runner /absolute/path/fixed-project-source/src/oh_my_duck/validation/asr_measurement.py \
  --expected-model Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output outputs/asr-measurement-NEW/independent-audit.json
```

三个入口均要求新的输出路径。测量按原始 Parquet 的完整数量执行，逐条
保存并刷新 `predictions.jsonl`；模型调用或输入检查失败时终止，已生成资料
保留。`result.json` 仅在完整语料全部完成后生成。

## 独立复核

`omd validate asr-audit` 根据完整的 FLEURS test 语料，核验实际转写记录、
原始录音和汇总指标。实现位于 `src/oh_my_duck/validation/asr_corpus.py`。
使用 `oh-my-duck[asr-evaluation]` 安装 Parquet、WAV 和识别指标依赖。
当前入口支持 CPU Float32 的 Qwen ASR 记录。

FLEURS 中文与英文完整 test 语料的 1,592 条实际录音已完成项目 ASR 推理与
独立复核，中文 CER 7.681%、英文 WER 5.306%。环境、全部耗时、
来源与数据拒绝检查见[真实录音测量](reports/asr-corpus-2026-10-12.md)。

```bash
omd validate asr-audit \
  --capture /absolute/path/asr-capture \
  --corpus /absolute/path/corpus-preparation/result.json \
  --source-root /absolute/path/recorded-project-source \
  --runner /absolute/path/recorded-measurement.py \
  --expected-model Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /absolute/path/independent-asr-audit.json
```

`--source-root` 指向测量时使用的固定 Git 源码；`--runner` 指向保存的实际
测量程序。输出文件必须使用新的路径。复核读取原始材料，不执行模型推理。

## 保存的数据

语料准备 JSON 包含 `passed`、`dataset`、`dataset_revision`、`split`、
`total_rows` 和 `splits`。每项 split 保存原始 Parquet 的绝对 `path`、
`sha256` 和完整 `rows` 数量。使用 Hugging Face 提供的固定 dataset
revision 与文件摘要检查下载来源。

测量目录包含 `result.json`、`predictions.jsonl` 和 `audio/`：

| File | Required evidence |
| --- | --- |
| `result.json` | 完成状态、项目 revision、Qwen 实现摘要、模型 revision、CPU Float32、语料身份、准备文件与逐条记录摘要、测量程序摘要、各语言汇总指标 |
| `predictions.jsonl` | 语言、连续 `row_index`、原始 `source_id`、原始路径及音频摘要、参考文字与实际转写、规范化文字、实际 WAV 路径及摘要、采样率、帧数、通道数、音频时长与转写耗时 |
| `audio/` | 每条录音对应独立的 Float WAV；解码后的全部采样值与原始语料一致 |

复核要求每项 split 覆盖全部记录，编号连续且没有重复。
所有音频文件均从实际语料重新解码，检查有限采样值、时长、通道和摘要。
参考文字与原始 Parquet 逐条对应。缺少记录、文件损坏、来源不符或指标不符
会在检查位置终止。

## 指标

文字规范化使用 Unicode NFKC、casefold、Unicode 标点移除和空白合并。
CER 计算前移除空白；`en_us` 同时计算 WER。识别指标由
[Jiwer](https://jitsi.github.io/jiwer/reference/process/) 计算，并保留 hits、
substitutions、deletions 和 insertions 数量。

耗时指标包括总音频时长、总转写耗时、real-time factor、中位数和 P95。
real-time factor 为总转写耗时除以总音频时长。
结果中的 `passed` 表示原始材料和测量结果一致；识别质量使用声明的错误率
及任务要求评估。

FLEURS 评估朗读录音的识别表现。机器人命令意图、噪声环境、机载麦克风和
完整交互延迟分别使用相应的实际样本与设备检查。
