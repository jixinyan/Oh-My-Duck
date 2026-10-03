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

`stop` 同时中断音频请求和原生任务，只有收到 native execution 的终止与 device confirmation 才发出 `voice.task.stop_confirmed` 和 `voice.stopped`。过期合成结果不会播放。物理制动检查由运动工具提供；执行中断记录原生控制边界和动作计数。设备、模型、传输与确认错误会终止会话并保留记录。

文件方式执行完整任务使用 `voice-task`；操作示例和实际结果见[上线验收](release-readiness.md)。2026-10-03 已完成录音文件、Qwen ASR、真实模型、Newton/BAM、正式目标验证和固定音色合成闭环。Microduck 音频设备尚待验证。

设备调用依据 [python-sounddevice Stream API](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html)；HTTP 调用依据 [HTTPX QuickStart](https://www.python-httpx.org/quickstart/)；服务请求与响应依据 [FastAPI Request](https://fastapi.tiangolo.com/advanced/using-request-directly/)及[自定义响应](https://fastapi.tiangolo.com/advanced/custom-response/)。
