# 原生目标接近验收

日期：2026-10-12。完整 CPU 任务使用 OpenAI `gpt-6-luna`、Responses、high 和
8192 tokens 输出预算，由固定 EDH 管理 planner、ActionGate 和独立 Verifier。
模型读取当前 `ball_0` 观测，调用 `microduck.approach_object`，完成接近、停止和正式任务。

## 运行来源

| 项目 | 固定来源 |
| --- | --- |
| 原生模型任务、工具权限与记录检查 | `c6ef937130ad799b32dda163ae8070bde0a79020` |
| 两组控制测量和中断检查 | `3ee612fc48b1c778bb29e2d608af2a8f91b4edb9`，控制器与原生会话内容保持一致 |
| 视频 renderer | `2e5a89e` |
| EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| 物理后端 | Linux CPU MuJoCo/BAM，OSMesa / llvmpipe，`CUDA_VISIBLE_DEVICES=''` |
| 模型连接主机 | macOS，凭证仅由私有配置读取到进程内存 |
| 场景 | 官方 8 × 6 米公寓，初始 `(-3.65, 0, 0)` |

目标接近使用当前 RGBD 观测的固定表面点，`standoff_m=0.4`。
原生任务保持物体中心半径 0.5 米、upright、五个保持控制步和 1000 控制步 / 300 秒预算。
感知来源为 `simulator_ground_truth`，保存原始目标与当前图像。

## 完整模型任务

| 测量 | 结果 |
| --- | --- |
| Run | `3a86de2b-cd74-45ff-b0a3-5207b2f22875` |
| 正式 verdict | `8083db06-2891-47ed-bfbf-24d3cd543c37`，`passed` |
| 任务状态与停止 | `succeeded`，确认的 `policy_stop` |
| 实际控制 / 物理子步 | 223 / 892 |
| 原始事件 / observer 帧 | 531 / 44 |
| 完成的模型 Responses | 37 |
| 距离误差 | 0.004715480830564556 米 |
| 停止记录检查 | 八项 progress，五个连续实际停止样本要求保持原样 |
| 资源释放 | 会话 `closed`、`released`；服务正常退出，所属 worker 数量为零 |

公开 `omd validate replay --require-target-approach` 从原始工具记录检查观测身份、
请求、表面点、实际位置、误差、停止和正式 verdict。该任务中每个接纳的目标请求均有
完成记录。完整记录位于 `outputs/acceptance/target-approach-model-mac-ssh-20261012-02/`。

## 控制、记录与安装

| 检查 | 结果 |
| --- | --- |
| Seed 20260929，初始 y=0、yaw=0 | 173 控制步，误差 0.004715480830564556 米，34 张 observer 图像 |
| Seed 20260930，初始 y=0.1、yaw=-0.55 | 184 控制步，误差 0.016358325879673674 米，36 张 observer 图像 |
| 执行中断 | 80 控制步时终止确认，结果为 `interrupted`，终止后 sequence 保持不变，资源释放 |
| 官方 ONNX 独立复核 | 逐动作检查 61 个观测、14 个动作和 13 个 command；上述三项最大 action 误差为零 |
| 原生请求接纳 | 缺少观测、未知目标、参数范围、待执行请求及旧观测的拒绝检查通过 |
| 原始记录的改动检查 | 观测、表面点、距离、sequence、停止计数、episode 六种改动，每组均被拒绝 |
| 原有前进与旋转 | 实际 0.4 米和 45 度回归，共 330 个控制步；误差、停止与资源释放检查通过 |
| 场景与导入 | 76 项检查通过，包含公共目标场景 |
| macOS 安装 | 449 个源码和资源文件、三份 license、31 项 CLI 调用通过 |
| Linux 安装 | 本机构建并安装 wheel / sdist，449 个文件、三份 license、31 项 CLI 调用及依赖检查通过 |

视频包含 observer、head RGB、公开 planner 文字、工具结果和正式 Verifier 记录。
1920 × 1080，H.264，10 fps，91.7 秒、917 帧；全部帧通过完整解码检查。
原始图像和视频保留 SHA256，画面时间与工具文本使用原始事件。

视频路径：`outputs/acceptance/target-approach-model-mac-ssh-20261012-02/agentic-target-approach-03.mp4`。
SHA256：`fbd3c08147438a60b4e2f327d2055e659d463db20fd015456ef7a4b40742fb99`。

汇交记录为 `outputs/acceptance/target-approach-handoff-20261012-01.json`；
独立几何检查为 `outputs/acceptance/target-approach-record-audit-20261012-01.json`；
安装检查为 `outputs/acceptance/target-approach-package-mac-20261012-01/result.json` 和
远程 `outputs/acceptance/target-approach-package-linux-20261012-01/result.json`。

这组验收覆盖静态可见目标的短距离接近。动态目标跟踪、模型感知来源的完整目标任务、
Newton 目标接近和真机执行各自保留待验证状态。GPU 验收与 RL 保持停止。
公共参数、场景与源码位置见[使用说明](../native-target-approach.md)。
