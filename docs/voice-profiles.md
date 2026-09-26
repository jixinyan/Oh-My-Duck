# 持久化音色资料

`SQLiteVoiceProfileStore` 位于 `oh_my_duck.voice.profiles`，安装 `oh-my-duck[voice]` 后可用。调用者在用户确认试听结果后创建 `VoiceProfile`，再调用 `save_confirmed(profile)`。确认动作由调用者负责；存储类保存已经确认的资料。

构造时传入专用本地目录，例如 `SQLiteVoiceProfileStore("/home/user/duck-voices")`。目录内的 `voice_profiles.sqlite3` 保存各 persona 的版本和活动版本，`audio/<sha256>.wav` 保存音频。调用 `active(persona_id)` 读取当前版本，调用 `get(persona_id, revision)` 与 `history(persona_id)` 读取旧版本。读取结果中的 `reference_audio.uri` 指向受管理的音频文件。应用程序应当长期保留整个目录，并为它设置适合参考声音资料的文件访问权限。

`save_confirmed` 要求 `revision` 从 1 开始，并对同一 persona 逐次增加 1。新版本资料与活动版本指针在同一次 SQLite 事务中提交。同一版本、相同音频 SHA256 与相同语义字段的重复请求不会改变活动版本；字段冲突会报错。各 persona 的版本独立。`persona_id` 仅用作数据库字段，不参与文件路径构造。音频按内容散列值保存，原始来源路径可以改变。

参考音频必须是已存在的本地绝对路径或本地 `file://` URI，并提供小写 SHA256 与 `audio/wav` 或 `audio/x-wav` 媒体类型。支持单声道和双声道 WAV，以及 `PCM_16`、`PCM_24`、`PCM_32`、`FLOAT` 编码。存储过程完整读取音频帧，检查采样值均为有限数值，核对提供的 SHA256，再保存受管理的副本。每次读取版本时重新核对受管理音频的 SHA256。网络 URI、缺失文件、损坏音频和不匹配的散列值都会报错。`voice_id`、参考文本、设计模型 revision、合成模型 revision 与带时区的 `confirmed_at` 均为必填字段。

真实音频验收使用 [Qwen3-TTS 官方 Base 示例](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py) 引用的 [参考音频](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav)。该文件是 24 kHz 单声道 Float WAV，SHA256 为 `480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c`。本地验收将文件保存在被 Git 忽略的 `.cache/voice-profiles/reference.wav`，测试可通过 `OH_MY_DUCK_VOICE_REFERENCE_WAV` 指定其他本地位置。公开仓库不包含这段音频。

## Qwen 推理与命令行

语音模块使用两个独立环境：`environments/voice-asr` 安装 `qwen-asr`，`environments/voice` 安装 `qwen-tts`。两个官方包固定的 `transformers` 版本不同，因此不能安装在同一个环境。两个环境通过 `--data-dir` 共用音色资料目录。

以下命令中的 `python` 分别指向对应环境中的 Python。录音输入为本地 WAV。`--device cuda:0` 指向 `CUDA_VISIBLE_DEVICES` 分配后的第一个可见 GPU。

```bash
CUDA_VISIBLE_DEVICES=5 uv run --project environments/voice-asr --locked python -m oh_my_duck.cli.voice \
  --data-dir .cache/voice-session --device cuda:0 transcribe recording.wav

CUDA_VISIBLE_DEVICES=5 uv run --project environments/voice --locked python -m oh_my_duck.cli.voice \
  --data-dir .cache/voice-session --device cuda:0 design \
  --description "温暖、清楚的中文声音" --reference-text "你好，我是小鸭。"

uv run --project environments/voice --locked python -m oh_my_duck.cli.voice \
  --data-dir .cache/voice-session confirm CANDIDATE_ID --persona duck-001

CUDA_VISIBLE_DEVICES=5 uv run --project environments/voice --locked python -m oh_my_duck.cli.voice \
  --data-dir .cache/voice-session --device cuda:0 synthesize \
  --persona duck-001 --text "你好，今天有什么计划？"
```

`design` 返回候选编号与试听 WAV 路径，不改变活动音色。试听后执行 `confirm`，把候选音频保存为活动音色的新版本；其后 `synthesize` 自动读取该版本。`active --persona duck-001` 可以查询保存的音色和模型 revision。模型通过 Hugging Face snapshot 下载并记录具体 commit SHA。确认时固定 Base 模型的 commit SHA，日常合成使用这个 revision。更换声音时再次生成候选并确认，旧版本留在资料目录中。

当前命令行处理已有录音文件。录音设备、播放设备和外部 Harness 接口仍需接入。
模型推理在工作线程中运行，同一个模型实例的调用由互斥锁串行执行。调用者取消等待后，已经启动的 GPU 推理仍会继续；播放中断功能需要音频设备实现。
独立语音 CLI 进程默认设置 `--cpu-threads 1`，也可明确指定正整数。直接使用模型适配器的服务进程应自行设置 PyTorch CPU 线程数量；适配器构造器不会修改进程级线程设置。
