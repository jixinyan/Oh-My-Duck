# CPU 语音办公室导航

录音指令经过 Qwen ASR，原文提交给 OpenAI Responses `gpt-6-luna` high；
原生 Harness 调用官方 policy、距离/角度工具、实际相机和距离传感器，
完成办公室目标并停止。独立 Verifier 返回 `passed`，随后 Qwen Base TTS
使用已确认的音色生成完成反馈。

## 来源与任务

| 内容 | 固定身份 |
| --- | --- |
| Runtime source | `e440bc19b098ca5d0bba28d1fde7ee3a8e1ea9cf` |
| Media source | `3f7f3c4e827c7bb1249c768dceb033277092bc95` |
| Native Harness | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| Official policy catalogue | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| Qwen/Qwen3-ASR-0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| Qwen/Qwen3-TTS-12Hz-0.6B-Base | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Run | `278aab2a-42a1-41df-a0e1-7eac0421f81b` |
| Session | `9c1dfcb4-dd95-49df-83f3-cf58f4079048` |
| Execution | `b59e6c8c-3714-401d-bc7b-a71060597ff4` |
| Episode | `e475a908f2084f588c03ad32b47a1c99` |
| Verifier verdict | `6fadfa89-16c1-4c45-9bcd-4fc14b7c0d88`, `passed` |

测试在 `jd_B300` 的 CPU MuJoCo/BAM 公寓运行，CUDA 不可见，使用软件渲染。
配置保留 seed `20260929`、spawn `(0,0,0)`、原生办公室范围
`x=[0.7,3.8], y=[-0.8,0.8]`、0.2 米 clearance、五个目标保持 tick、
4000 个控制步和 1800 秒预算。Policy 保留官方 61 维观测、14 个 servo、
HOME、50 Hz 和每次控制四个 0.005 秒物理子步。原生 ActionGate、
Planner 与独立 Verifier 管理实际执行。

模型调用读取用户授权的 `decision-api-robot-demo/config/config.toml`，密钥仅在
进程内传递。模型检查与原生任务使用同一 provider 配置和环境代理。
实际模型检查返回 `OK`，任务的模型身份记录为 `gpt-6-luna`。
完整事件、model transport audit 和进程记录保留在独立输出目录中。

输入由已确认音色的真实 Qwen Base TTS 生成，Qwen ASR 识别为：

> 请使用距离和角度工具前往办公室。读取相机和距离传感器，到达后停止。

读取场景时明确使用 `simulator_ground_truth`。实际 head RGB、ToF、机器人状态
和当前物理 sequence 共同提供任务证据。

## 测量与结束

| 完成的请求 | 实际测量 | 误差 | 停止样本 |
| --- | --- | --- | --- |
| Walk +0.2 m | 0.184464 m | 0.015547 m | 5 |
| Rotate +30° | 33.800961° | 3.800961° | 5 |
| Rotate +10° | 5.171251° | 4.828749° | 5 |
| Walk +0.5 m | 0.507385 m | 0.027698 m | 5 |
| Walk +0.1 m | 0.080332 m | 0.020385 m | 5 |

旋转 +30° 的实际平移为 0.123754 米，旋转 +10° 的实际平移为
0.011478 米。Planner 根据当前测量继续规划。各项误差使用原有距离和角度
完成条件检查；未执行的准备请求保持取消身份。

本次执行包含 666 个 policy 控制步、2664 个物理子步和 1323 个事件。
独立 replay audit 检查 133 个 observer 帧及 17 次停止进度，全部 141 张原始
PNG 通过 SHA256、尺寸和图像内容检查。最后的零命令保持执行 50 个控制步，
最终 sequence 为 666，连续停止样本为 41，原生目标保持为 106/5。
最终 body position 为 `(0.747317,0.454169,0.116770)` 米，tilt 为
`0.005673` rad。终止原因为设备确认的 `policy_stop`，原生 run 为 `succeeded`。

## 语音资料与资源释放

Persona 为 `语音验收小鸭`，voice 为 `已确认音色一号`，profile revision 为 1。
Profile 内容、参考音频和数据库字节保持一致。

| 音频 | 时长 | PCM frames | SHA256 |
| --- | --- | --- | --- |
| 指令 | 5.76 秒 | 138240 | `28dfc0e46b3fa2b3c5b771d15d5f8e3929f3156d5f65e754e16bcec1f1f3d8f2` |
| 完成反馈 | 1.04 秒 | 24960 | `11217414313a4be025bd86d4f19940e76943e18be75f85b91639a24e1f54d3b1` |

两段原始 WAV 均为 PCM16、24 kHz、mono，通过完整读取、有限值和非零内容检查。
本次使用文件输入，`live_microphone=false`，`speaker_playback=false`。
Model check、Harness、voice-task、ASR 和 TTS 五个所属进程均以 exit code 0
结束。会话关闭并释放资源，端口 19861、19862、19863 关闭；独立生命周期检查
确认全部记录对应的进程已经退出。

## 安装与媒体检查

Runtime source 的 wheel、source distribution 和项目目录之外的独立安装通过
433 个源码/资源文件、三份许可证和 26 个实际 CLI 调用。检查包含保留的
1208 个真实 ONNX policy 输出和连续五动作的独立物理资料复核。
媒体入口使用真实保留资料验证 task/run 身份与 WAV SHA256，两个不匹配输入
均被拒绝，原始视频和音频保持一致。

完成的 agentic MP4 为 H264、1920×1080、10 fps、1745 帧、174.54 秒，
音轨为 AAC、24 kHz、mono。画面包含实际 observer、head RGB、工具请求、
物理进度与正式 Verifier 结果；视频呈现到 `run.succeeded` 的第 1311 个事件，
全部 1323 个原始事件保留在 replay 中。全部编码画面的文字范围检查、
完整视频与音轨解码、媒体尺寸、帧数和 SHA256 检查通过。

指令音频从视频开头播放，原生任务画面从 6.3 秒开始；166.7 秒原生视频结束后，
反馈从 173 秒开始播放，最终画面保持 1.54 秒。等待画面按 8 倍墙钟时间呈现，
运动段保留实际 simulator 时间下限，133 个 observer 时间节点检查通过。
原始音频、相机图像与任务记录保持一致。

最终视频为 `outputs/demos/cpu-luna-voice-office-20261008-07.mp4`，SHA256 为
`7cab2c748ccc9670a35678569bbc0cef25983498cc567c6caf19b4afe3de0fdd`。
相邻 `.voice.json` 保存任务、视频、音频与处理脚本来源；独立媒体检查保存于
`outputs/acceptance/cpu-voice-video-20261008-07-02/result.json`。

## 复核与生成视频

使用独立输出路径，保留原始任务记录。通过已安装的 CLI 复核 replay：

```bash
omd validate replay outputs/acceptance/cpu-voice-loop-20261008-07/replay \
  --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-metric-tools
```

使用锁定的 Demo 环境生成实际事件与相机画面，再加入同一任务的原始音频：

```bash
environments/demo/.venv/bin/python scripts/render_microduck_run.py \
  --export outputs/acceptance/cpu-voice-loop-20261008-07/replay \
  --output outputs/demos/voice-office-new-silent.mp4 \
  --review-dir outputs/demos/voice-office-new-review --fps 10 --wall-speed 8
environments/demo/.venv/bin/python scripts/add_voice_to_demo.py \
  --video outputs/demos/voice-office-new-silent.mp4 \
  --voice-result outputs/acceptance/cpu-voice-loop-20261008-07/task/result.json \
  --instruction-audio outputs/acceptance/cpu-voice-loop-20261008-07/input/0dee47142b6242cbab664112ddce5b19.wav \
  --response-audio outputs/acceptance/cpu-voice-loop-20261008-07/task/speech/55a74a9338c94ea990b770f6637e389b.wav \
  --output outputs/demos/voice-office-new.mp4
```

原生任务的场景、目标、policy 和模型分别通过
[场景配置](../harness-native-integration.md)、[voice-task](../voice-interaction.md)
和[发布流程](../release-readiness.md)进入相同接口。

## 保存的证据

- `outputs/acceptance/cpu-voice-loop-20261008-07/result.json`
- `outputs/acceptance/cpu-voice-loop-20261008-07/task/result.json`
- `outputs/acceptance/cpu-voice-loop-20261008-07/replay/`
- `outputs/acceptance/cpu-voice-loop-20261008-07/audit.log`
- `outputs/acceptance/cpu-voice-loop-20261008-07/independent-lifecycle-audit.json`
- `outputs/acceptance/cpu-voice-loop-20261008-07/remote-cleanup.json`
- `outputs/acceptance/voice-media-20261008-03-install/result.json`
- `outputs/acceptance/voice-media-admission-20261008-03/result.json`
- `outputs/acceptance/cpu-voice-video-20261008-07-02/result.json`
- `outputs/demos/cpu-luna-voice-office-20261008-07.mp4`
- `outputs/demos/cpu-luna-voice-office-20261008-07.voice.json`

GPU 验收和 RL 保持停止。当前 Newton 运行、多场景长导航、图像输入独立 VLN、
模型识别准确率、学习行为和 Microduck 音频设备需要各自的实际验收。
