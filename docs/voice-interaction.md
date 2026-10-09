# 显式录音与语音播放

`voice-session` 在本机操作 PortAudio 输入和输出设备，使用 JSONL 接受命令并输出 `EpisodeEvent`。ASR 与 TTS 分别运行在独立的 Python 环境中，以 FastAPI 提供 HTTP 服务。客户端使用 HTTPX 的异步连接。远端服务只绑定 `127.0.0.1`；跨主机连接使用 SSH 本地端口转发。

客户端环境为 `environments/voice-client`，包含 `sounddevice`、`soundfile`、NumPy 和 HTTPX。ASR 环境为 `environments/voice-asr`，TTS 环境为 `environments/voice`。在各自主机的项目根目录安装锁定依赖，并将临时文件放在被 Git 忽略的项目目录：

```bash
mkdir -p .cache/tmp
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice-client --locked --no-editable
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice-asr --frozen
TMPDIR="$PWD/.cache/tmp" uv sync --project environments/voice --frozen
```

本机只需安装 client，远端分别安装 ASR 与 TTS。`omd.py` 会把项目的 `src` 目录加入 Python 搜索路径；独立验收脚本运行时显式设置 `PYTHONPATH=src`。模型服务启动时加载一次明确指定的模型 commit，使用一个 Uvicorn worker。指定物理 GPU 时设置 `CUDA_VISIBLE_DEVICES`，服务参数使用映射后的 `--device cuda:0`。音色设计及确认沿用[音色资料说明](voice-profiles.md)。TTS 服务的 `--data-dir` 必须是已确认音色资料目录；每次请求调用 `SQLiteVoiceProfileStore.active(persona_id)` 并核对该版本保存的 Base 模型 revision。

在远端对应环境中启动两个服务，以下端口均绑定远端本机。先用 `nvidia-smi` 核对当前空闲设备，并把 `GPU_ID` 设置成获得使用许可的物理 GPU 编号。2026-09-29 的 GPU5 验收服务已经关闭，GPU5 随后分配给 RL 训练；下面的命令不表示该设备目前可用。

```bash
TMPDIR="$PWD/.cache/tmp" CUDA_VISIBLE_DEVICES="$GPU_ID" environments/voice-asr/.venv/bin/python omd.py voice-service asr \
  --data-dir .cache/voice-runtime/asr \
  --model-revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 --device cuda:0 --port 18761

TMPDIR="$PWD/.cache/tmp" CUDA_VISIBLE_DEVICES="$GPU_ID" environments/voice/.venv/bin/python omd.py voice-service tts \
  --data-dir .cache/voice-runtime/profiles \
  --model-revision 5d83992436eae1d760afd27aff78a71d676296fc --device cuda:0 --port 18762
```

客户端所在主机建立 SSH 本地端口转发：

```bash
ssh -N -L 18761:127.0.0.1:18761 -L 18762:127.0.0.1:18762 jd_B300
```

查询 `sounddevice.query_devices()` 后，明确选择实际麦克风与扬声器。运行会话时提供 persona、机器人身份和执行领域，并把运行记录写入选择的 JSONL 文件：

```bash
TMPDIR="$PWD/.cache/tmp" environments/voice-client/.venv/bin/python omd.py voice-session \
  --persona '语音验收小鸭' --robot-id mac-audio --domain real \
  --input-device 'MacBook Pro Microphone' --output-device 'MacBook Pro Speakers' \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --data-dir .cache/voice-runtime/client --log .cache/voice-runtime/events.jsonl
```

输入的每一行都是一个 JSON object，包含唯一 `request_id` 和 `command`。`begin_recording` 返回 `recording_id`；`finish_recording` 使用该编号结束采集，返回本地 WAV URI 和 SHA256。`transcribe` 默认处理最近完成的录音，也可以提供绝对路径 `audio`；结果包含文本、ASR 模型 revision 和录音 SHA256。调用者可以把该文本交给外部 Harness，再使用 `speak` 提交明确的回复文本。`speak` 包含 `text`，结果事件包含音色版本、模型 revision、WAV SHA256 与播放编号。`stop` 取消正在等待的请求，停止实际播放；录音正在进行时同时结束录音并返回 WAV。`status` 报告设备的当前编号、运行情况和保留的错误。

```json
{"request_id":"1","command":"begin_recording"}
{"request_id":"2","command":"finish_recording","recording_id":"RECORDING_ID"}
{"request_id":"3","command":"transcribe"}
{"request_id":"4","command":"speak","text":"你好，我们开始吧。"}
{"request_id":"5","command":"stop"}
{"request_id":"6","command":"status"}
```

长请求先发出 `voice.request.accepted`，完成后发出转写或播放事件。每个事件包含 session、episode、request 身份与 UTC 时间。`voice.playback.completed` 表示 PortAudio 播放自然完成；`voice.stopped` 在设备停止确认之后发出。录音启动之前会停止当前播放，录音期间的播放请求会报错。中断合成时，即使远端 GPU 已开始推理，过期请求也不会自动播放其结果。音频设备或模型服务错误会写入事件并终止会话，保留现场错误供调用者处理。

## 原生 Harness 任务

`voice-session` 接受完整的 `--harness-url`、`--harness-profile` 和 `--harness-scenario` 参数。打开会话时核对原生 session 的执行领域；`--domain simulation` 要求 simulation，`--domain real` 要求 hardware。服务地址使用本机回环地址。

```bash
environments/voice-client/.venv/bin/python omd.py voice-session \
  --persona '语音验收小鸭' --robot-id microduck --domain simulation \
  --input-device 'MacBook Pro Microphone' --output-device 'MacBook Pro Speakers' \
  --harness-url http://127.0.0.1:4318 \
  --harness-profile nvidia-office-6.0 --harness-scenario navigate-nvidia-office-6.0 \
  --data-dir outputs/voice/session-new --log outputs/voice/session-new/events.jsonl
```

完成录音后发送 `execute_recording`，也可以通过绝对路径指定已有 WAV：

```json
{"request_id":"task-1","command":"execute_recording","audio":"/absolute/path/command.wav","language_hint":"Chinese"}
{"request_id":"stop-1","command":"stop"}
```

`execute_recording` 将 ASR 文本原样传入原生任务，记录 catalogue digest、run ID、正式 verdict 和音色版本；任务完成后使用已确认音色播放状态反馈。原生 Harness 负责规划、工具选择、执行权限、记忆与独立 Verifier。录音和播音均属于 Microduck 语音模块。

后续命令可以通过 `context_run_ids` 明确选择同一原生 session 中最多四项已结束任务。
编号来自 `voice.task.started` 的 `runId`，当前 session/run 身份也通过 `status` 返回。
原生服务核查任务归属和终止状态，将历史指令、结果、正式验证结论与 skill ID
写入新任务的 `history_summary`。新任务使用当前世界物理状态和实际传感器。

```json
{"request_id":"task-2","command":"execute_recording","audio":"/absolute/path/followup.wav","context_run_ids":["FIRST_RUN_ID"]}
```

`context_run_ids` 必须是数组，编号为 1–80 个 ASCII 字母、数字或连字符，最多四个且不能重复。
格式错误在 ASR 请求之前终止；跨会话引用和正在执行的任务由原生服务拒绝。
原生历史摘要编码上限为 16 KiB；服务在接纳位置检查完整摘要。
`transcribe` 接收 `context_run_ids` 时终止请求。
连续提交等待原生 session 完成任务收尾并返回 `ready`；等待上限为 240 秒，
身份、来源、资源状态与收尾错误在读取位置终止。
关闭会话要求原生服务确认 `closed` 与 `resources=released`。

每项原生任务创建独立的 `goal_scope_id`，记录 `goal_bound_sequence`，并从零累计
`held_ticks`。任务绑定保留 episode、位姿、速度、当前 policy 和上一动作；目标条件
继续使用场景原有定义。读取当前传感器和目标状态保留保持次数，新的实际 policy
控制步才能增加计数。到达后的观察任务需要读取当前传感器、执行零 twist 保持，
取得本任务的目标与停止证据。Newton 路线中的 waypoint 记录也按任务重新开始。

`omd harness --max-output-tokens 8192` 明确设置模型输出预算，包含 reasoning tokens。
默认值为 4096，接受范围为 256–8192；原生 context compaction 保留对应输出空间。
模型没有完成输出时保留原生错误与停止边界。

原生 compaction 管理历史摘要。模型适配代码通过 `purpose=compaction`
说明辅助请求的作用范围：摘要保存原有任务指令、待执行动作与验证要求。
当前用户指令、工具权限和实际执行状态继续由原生任务记录管理。
实现位于仓库根目录的 `integrations/edh/model-context.mjs`。

文件方式也能在同一会话内依次执行多个真实录音。`--context-previous-task`
明确让每项后续任务引用上一项成功任务；该参数需要至少两个输入文件。

```bash
omd voice-task --audio /absolute/path/command.wav /absolute/path/followup.wav \
  --context-previous-task --persona '语音验收小鸭' \
  --harness-url http://127.0.0.1:4318 \
  --profile official-apartment-office --scenario navigate-office \
  --asr-url http://127.0.0.1:18761 --tts-url http://127.0.0.1:18762 \
  --language-hint Chinese --output outputs/voice/sequence-new
```

每项任务分别保存转写、引用、提交、原生 run 和固定音色反馈。
任务失败或传输错误终止文件序列并关闭会话。`result.json` 的 `tasks` 保存完整序列，
顶层任务字段记录最近完成的任务，单文件调用继续支持原有字段。
跨会话经验通过原生 `skills.search` 和 `skills.load` 按需读取。
已关闭会话及服务重启后的历史会话保持只读。

`stop` 同时中断音频请求和原生任务，只有收到 native execution 的终止与 device confirmation 才发出 `voice.task.stop_confirmed` 和 `voice.stopped`。过期合成结果不会播放。物理制动检查由运动工具提供；执行中断记录原生控制边界和动作计数。设备、模型、传输与确认错误会终止会话并保留记录。

文件方式执行完整任务使用 `voice-task`；操作示例和实际结果见[上线验收](release-readiness.md)。2026-10-03 已完成录音文件、Qwen ASR、真实模型、Newton/BAM、正式目标验证和固定音色合成闭环。Microduck 音频设备尚待验证。

设备调用依据 [python-sounddevice Stream API](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html)；HTTP 调用依据 [HTTPX QuickStart](https://www.python-httpx.org/quickstart/)；服务请求与响应依据 [FastAPI Request](https://fastapi.tiangolo.com/advanced/using-request-directly/)及[自定义响应](https://fastapi.tiangolo.com/advanced/custom-response/)。
