# 语音交互验收 · 2026-09-29

本次验收使用 `feat/interactive-voice-20260929` 工作树，基线为 `caabb4e820b49a88808eafc498a2a9c4fd21abb0`。Mac 使用真实 `MacBook Pro Microphone` 和 `MacBook Pro Speakers`；远端 `jd_B300` 在独立 worktree 中运行 ASR 与 TTS，两个服务均只绑定远端 `127.0.0.1`，通过 SSH 本地端口转发连接。ASR、声学采集及推理期间中断的模型推理指定物理 GPU5。两项 GPU 服务进程均已退出；结束检查时 GPU5 为 1 MiB、0% 使用率，随后该设备交由 RL 训练使用。会话正常播放及播放后中断的补充验收使用远端 CPU 模型服务。

ASR 使用 `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`；TTS 使用 `Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc`。两个模型分别运行于锁定的 `environments/voice-asr` 与 `environments/voice`，本机设备与 HTTP 客户端运行于 `environments/voice-client`。TTS 服务从独立验收资料目录读取 `语音验收小鸭` 的活动版本 1，其 `voice_id` 为 `已确认音色一号`，参考音频 SHA256 为 `e70207c688d248a1b51feddf3661586abdeb830d255d68d46e49353ff24f3861`。该资料由既有真实 Qwen VoiceDesign 参考音频写入独立验证 persona，未更改其他 persona 的活动版本。未确认的 persona 请求返回 HTTP 404。

保存的[完整验收脚本](../../scripts/validate_voice_interaction.py)在同一轮中执行 HTTP 合成、HTTP 转写、真实播放中断与自然完成、独立扬声器进程到麦克风的声学采集，以及远端推理期间的会话中断。客户端结果保存于 `.cache/voice-full-validation-20260929-1/result.json`，原始会话事件保存于 `.cache/voice-full-validation-20260929-1/session/events.jsonl`；远端服务日志与生成音频保存于独立 worktree 的 `.cache/voice-validation/`。本地原始 WAV、结果和事件都保留在同级 `Oh-My-Duck-voice` 工作树的 `.cache` 中；主 checkout 中相同相对路径没有这些测试文件。

合成输入为“你好，我是小鸭。我们现在检查语音连接。”。TTS 返回 24 kHz、单声道、3.44 秒 WAV，SHA256 为 `3c8f4e008c39cbb58e218b66296ba5e5e5f5ef83577b789b4e89df315b12a923`，响应 metadata 包含中文 persona、中文 voice ID、活动版本 1 和固定 Base 模型 revision。该文件直接通过 HTTP 发送到 ASR 后，回读为“您好，我是小丫。我们现在检查语音连接。”；ASR 返回同一音频 SHA256 与固定模型 revision。

随后，独立进程通过 `MacBook Pro Speakers` 播放该 Qwen WAV，`VoiceSession` 同时通过 `MacBook Pro Microphone` 采集 187392 帧、48 kHz 的真实录音。录音 RMS 为 0.0332816653，峰值为 0.3534851074，SHA256 为 `2565e3bd1f483b665fd80e96350013042622bbb8b1954eac12a99f5f7ef58309`。录音经远端 ASR 回读为“您好，我是小丫。我们现在检查语音连接。”。该请求从 `voice.request.accepted` 到 `voice.transcription.completed` 经过约 7.29 秒，记录的是本轮端到端耗时；实时响应性能尚未验收。合成原文的“你好／小鸭”与回读的“您好／小丫”分别保留；本次验证证明真实设备采集、音频传输与语音识别路径可运行，没有测量语音识别准确率。输入音源为真实 Qwen 合成语音经扬声器播出，没有真人发话。

播放中断由 PortAudio 返回 `stopped`，自然播放结束返回 `completed`。对连续 5 次播放后立即开始录音及连续 10 次无等待的录音开始／结束进行了真实设备检查，没有出现未知句柄或设备错误；极短录音仅用于状态检查。

补充会话播放验收使用同一固定 Base 模型 revision、同一独立已确认 persona，在远端显式设置 `--device cpu` 与空 `CUDA_VISIBLE_DEVICES`。CPU 服务 PID 525775 的 `/health` 返回固定模型 revision，服务日志包含两次实际 `POST /v1/speech` 的 HTTP 200；`nvidia-smi` 的计算进程列表没有该 PID。保存的 `.cache/voice-session-cpu-20260929-1/session/events.jsonl` 记录 `complete-speak` 的 `voice.playback.started`（23:47:43.838908 UTC）与同一句柄的 `voice.playback.completed`（23:47:47.946580 UTC），WAV SHA256 为 `9c49bd26a24d6441b446764036fb02ac9d5397d2c5b4c537a12c25d458c4119e`。随后 `stop-speak` 的实际 `voice.playback.started` 发生于 23:48:25.387926 UTC；`playback-stop` 在 23:48:25.495224 UTC 返回同一句柄的 `playback_outcome=stopped`，后续状态 `playback_active=false`，该请求没有 `voice.playback.completed`。两次播放都使用真实 `MacBook Pro Speakers`。结果保存于 `.cache/voice-session-cpu-20260929-1/result.json`，脚本退出码为 0。CPU 服务已经关闭，PID 525775 与本地 SSH 端口监听均已消失。

在真实 TTS 服务日志记录 `voice.speech.started` 后，`VoiceSession` 收到 `stop` 并记录原请求的 `voice.request.cancelled`。远端工作线程随后生成 WAV `1b9c1bf9666f4c0b94b7503690583d70.wav`；会话状态报告 `playback_id=null`、`playback_active=false`，事件日志中没有该请求的 `voice.playback.started`。这项证据表明客户端中断阻止了迟到播放，远端已开始的模型推理仍会执行完毕。

运行命令如下；开始前需按[音色资料说明](../voice-profiles.md)准备独立已确认 persona，并启动[语音服务与 SSH 隧道](../voice-interaction.md)。`--output-dir` 使用每次新建的被 Git 忽略目录，避免覆盖旧证据。

```bash
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src environments/voice-client/.venv/bin/python scripts/validate_voice_interaction.py \
  --persona '语音验收小鸭' --robot-id mac-audio \
  --input-device 'MacBook Pro Microphone' --output-device 'MacBook Pro Speakers' \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --ssh-host jd_B300 \
  --remote-tts-log /mnt/data/users/jixin/workspace/code/Oh-My-Duck/.cache/voice-worktrees/voice-20260929/.cache/voice-validation/tts.log \
  --remote-tts-output /mnt/data/users/jixin/workspace/code/Oh-My-Duck/.cache/voice-worktrees/voice-20260929/.cache/voice-validation/profile-store/output \
  --output-dir .cache/voice-full-validation-20260929-2
```

保存的完整验收运行退出码为 0，结果目录为 `.cache/voice-full-validation-20260929-1`；上述命令使用新的目录供重复执行。`tests/test_voice_profiles.py` 使用官方真实参考 WAV 返回 `10 passed`；三个语音环境的 `uv lock --check`、Python 编译检查、`git diff --check` 均通过。本机 `environments/voice-client/.venv/bin/python omd.py voice-session --help`，远端 `environments/voice-asr/.venv/bin/python omd.py voice --help` 及 `environments/voice/.venv/bin/python omd.py voice-service --help` 均通过。会话将转写文本交给外部调用者，接收调用者提交的合成文本；外部 Harness 的真实联接仍由其接口提供。

补充会话播放验收使用保存脚本的 `session-playback` 模式。远端项目目录为 `/mnt/data/users/jixin/workspace/code/Oh-My-Duck/.cache/voice-worktrees/voice-20260929`。在该目录启动真实 CPU TTS 服务，其固定模型文件来自项目 `.cache/voice-hf`；服务启动后在本机建立 `ssh -N -L 18762:127.0.0.1:18762 jd_B300` 隧道，再在本机语音工作树运行脚本：

```bash
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" HF_HOME=/mnt/data/users/jixin/workspace/code/Oh-My-Duck/.cache/voice-hf \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= environments/voice/.venv/bin/python omd.py voice-service tts \
  --data-dir .cache/voice-validation/profile-store \
  --model-revision 5d83992436eae1d760afd27aff78a71d676296fc --device cpu --port 18762
```

```bash
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src environments/voice-client/.venv/bin/python scripts/validate_voice_interaction.py \
  --mode session-playback --persona '语音验收小鸭' --robot-id mac-audio \
  --input-device 'MacBook Pro Microphone' --output-device 'MacBook Pro Speakers' \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --output-dir .cache/voice-session-cpu-20260929-2
```

该模式只调用 TTS；`--asr-url` 保留会话配置值。每次运行使用新的 `--output-dir`。远端 CPU 服务日志位于 `.cache/voice-validation/tts-cpu.log`。模型推理由远端服务完成，Mac 负责真实扬声器播放。
