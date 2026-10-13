# 模型感知驱动的目标任务

CPU MuJoCo/BAM 公寓中的实际任务使用 `gpt-6-luna`、YOLO26、官方 ONNX policy、
原生 EDH ActionGate 和独立 Verifier。Agent 根据检测结果选择目标，在不同物理
sequence 查询同一个 track，调用 `approach_object`，完成停止并取得正式 `passed` verdict。

## 来源

| 项目 | 实际来源 |
| --- | --- |
| 任务与记录检查源码 | `ed1f3e93145b7614fabd1d5e1f1af857041545ee` |
| 视频 renderer 源码 | `498e8e5`，SHA256 `103426b21794847614d861eec7ccbe2fce388bc2e1323a4c2eed57276e8d58f6` |
| 原生 EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy catalogue | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| YOLO26 checkpoint | SHA256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| Brain | `gpt-6-luna`、Responses、high、8192 output tokens |
| 物理执行 | Linux CPU MuJoCo/BAM、OSMesa；Mac 原生 Harness 通过 SSH 使用 worker |
| 场景 | `configs/simulation-demo/apartment-model-target.json`，seed `20260929` |
| Run | `55d378cb-57c0-479f-abf0-15880b35cdd1` |
| Verdict | `beb74e25-3781-4874-94f8-5a7386a3ecfa` |

private provider 配置由用户授权，凭据在进程内读取。模型服务的初始和最终状态均为
`device=cpu`、YOLO 参数全部位于 CPU、`cuda_initialized=false`。CPU worker 的
`CUDA_VISIBLE_DEVICES` 为空。GPU 验收和 RL 保持停止。

## 实际任务

场景起始位置为 `(-1.6, -0.5, π)`。Agent 使用 `source=models`、`prompt=objects`，
从当前 head RGB 选择 RGBD 有效的小型目标。三次模型观测的物理 sequence 为
75、85、95；相同 `track_id` 在 sequence 85 和 95 查询得到 `status=current`。

本次模型返回的原始标签为 `orange`，用于接近的目标置信度为 `0.5196300745010376`。
记录和视频保留这个标签。该任务验证检测、关联、测量与控制流程；类别识别准确率
需要独立标注数据和评估。距离来自仿真 RGBD。

接近请求使用 sequence 95 的原始 `target_id`，水平表面点为
`(-2.9766592979431152, -0.006067246198654175, 0.03531990200281143)`。
请求保持距离 0.4 米，动作中保存该固定位置、原始检测和模型 SHA256。
`alpha_walking` 根据实际身体位置和 bearing 执行控制。完整任务的结果为：

| 测量 | 结果 |
| --- | --- |
| 正式任务状态 | `succeeded` |
| 独立 Verifier | `passed` |
| 停止原因 | `policy_stop`，设备确认 |
| 控制步 / 物理子步 | 496 / 1,984 |
| 原始导出事件 | 965 |
| Observer 图像 | 99 张，640 × 480 |
| 完成的 Responses | 51 |
| 接近请求 | 一个，来源为 `models` |
| 水平表面距离误差 | `0.0006718723982021557` 米；允许 0.05 米 |
| 最终身体位置 | `(-2.612376891416987, -0.17287245686016836, 0.11666026615269467)` |
| 身体到原生 `ball_0` 中心距离 | `0.42442100973320757` 米；范围 0.5 米 |
| 连续停止样本 | 55；要求五个 |
| 原生目标保持 | 102 个控制步；要求五个 |
| 高度 / 倾斜 | `0.11666026615269467` 米 / `0.007084301009823894` rad |
| Session 关闭 | `closed`、`resources=released` |
| 退出后 CPU worker | 零个 |
| 感知服务 | 退出状态为零，8784 端口停止监听 |

## 原始资料与独立检查

每次服务请求保存原始 `frame.json`、`response.json` 和 SHA256。独立检查将三次模型
输出与 Agent 工具收到的目标、图像 attachment、episode、sequence、时间、query 和
模型 SHA256 关联。原始 PNG 字节与实际工具图像相同。

公开 `omd validate perception --capture` 格式的资料保存于独立检查目录，包含原始帧、
模型响应、真实 health 和 capture manifest。检查程序直接从原始 RGBD 像素重新计算
距离、分位数、world 表面点和 bearing。三次 capture 与三个模型目标通过，
54 项修改后的测量被拒绝。七项修改后的来源或模型身份被目标请求检查拒绝。

独立安装的 `omd validate replay --require-model-perception --require-target-tracking`
在项目目录之外检查正式结果、实际位置、原始观测、两个当前查询和停止状态。
Mac 与 Linux 的 wheel、源码包和安装结果分别通过 454 个文件、三份 license、
33 项公开命令检查。80 项场景 Schema 与导入检查通过。

## MP4

`outputs/acceptance/model-target-mac-ssh-20261013-02/agentic-model-target-01.mp4`
包含实际 observer、模型标注后的 head RGB、公开 Planner 输出、工具参数、目标查询、
原始检测标签、实际运动和独立 Verifier。

| 检查 | 结果 |
| --- | --- |
| 编码 | H.264，1920 × 1080，10 fps |
| 帧数 / 时长 | 1,600 / 160 秒 |
| SHA256 | `fe75a6c93f35ac195a4163542153e42042148568da4b526cf3065d7409df29a1` |
| 完整解码 | 1,600 帧全部通过 |
| 每帧文字范围 | 全部通过 |
| 图片检查 | 五张关键帧通过 PNG 解码、尺寸与数值变化检查 |
| 原始资料 | manifest、run、events、renderer SHA256 通过复核 |
| 公开 trace | Planner 文本与工具结果；记录中 reasoning events 为零 |

## 产物

| 内容 | 位置 |
| --- | --- |
| 完整原生任务与视频 | `outputs/acceptance/model-target-mac-ssh-20261013-02/` |
| 原始服务 capture 与退出记录 | `outputs/acceptance/model-target-perception-linux-20261013-02/` |
| 原始 RGBD 与来源修改检查 | `outputs/acceptance/model-target-rgbd-audit-mac-20261013-01/result.json` |
| 安装版本的正式任务复核 | `outputs/acceptance/model-target-installed-replay-mac-20261013-01.json` |
| Mac 安装检查 | `outputs/acceptance/model-target-package-mac-20261013-03/result.json` |
| Linux 安装检查 | `outputs/acceptance/model-target-package-linux-20261013-01.json` |
| Schema 与导入检查 | `outputs/acceptance/model-target-imports-20261013-03.xml` |
| 独立资料复核 | `outputs/acceptance/model-target-handoff-20261013-01.json` |

当前通过范围为一个固定 CPU 场景的完整模型感知任务。自主跟随、物体交叉、长时间遮挡、
类别识别准确率、多场景重复模型任务、当前源码的 Newton GPU 执行和真机执行具有独立验收要求。
