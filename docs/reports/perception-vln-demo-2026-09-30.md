# 感知工具、多阶段导航与 MP4 · 2026-09-30

`microduck.inspect_scene(prompt, source)` 已接入原生 Harness。Newton Office 的真实模型任务完成大厅中心、右侧绕行和办公桌接近；独立 Verifier 为 `passed`，原生任务为 `succeeded`。任务明确使用 simulator ground truth 的实例分割、射线距离、bearing、公共几何与 world odometry。当前行为范围为单一 Office 场景、seed 20260929。

## 感知

独立环境为 `environments/perception`，锁文件使用官方 headless Ultralytics 8.4.170、Torch 2.10.0/CUDA 13 和 SAM 源码 `2345a4ad109ac29c569da749c91d84f10dc08c40`。YOLO26s checkpoint SHA256 为 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`。

实际 Newton 的两个 head camera 朝向共得到 21 个有效 ground truth 目标。YOLO26 在同一 RGB/depth 输入上完成实际 GPU 推理：朝向东方返回四项候选，朝向办公桌没有模型目标。该检查证明推理与 calibrated depth 数据传输，识别准确率尚未验收。[原始结果](evidence/perception-newton-20260930.json)保留每项类别、置信度、距离、bearing、来源和模型 SHA256。

SAM 3.1 适配使用官方 `build_sam3_predictor(version="sam3.1")`、文本分割与 YOLO 框关联。服务器当前 Hugging Face 账号访问官方 checkpoint 返回 403，作者提示访问申请被拒绝，联合推理尚未验收。使用说明见[感知与导航工具](../perception-navigation.md)。

感知距离取当前可见目标像素到光心的距离中位数，并报告 10%–90% 区间；水平距离取身体到可见表面中位数位置的距离。桌前停靠距离另使用身体位置到 authored static USD geometry bounds 的距离，两项测量具有各自的含义。

## 实际任务

运行身份为 `71dac4ee-298c-4895-b3db-a9610bce50bb`，profile 为 `nvidia-office-vln`。机器人执行官方 `velstand` 与 `alpha_walking`，通过固定 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd` 的 SessionEnvironment、ActionGate 和独立 Verifier。模型连接为 Responses / `gpt-6-astra` / high，官方 policy revision 为 `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。

| 项目 | 实际结果 |
|---|---|
| 控制与物理 | 3689 个控制步，14756 个 Newton/BAM 子步；50 Hz、四个 0.005 秒子步 |
| 模型工具调用 | 3 次 walk，7 次 rotate；其中一次转向请求因当前运动危险状态被拒绝 |
| 办公桌观测 | 7 次包含桌子的不同物理帧；全部标记 simulator ground truth |
| 行走段累计测量 | 4.166466 m，使用各段实际 odometry 投影；包含返回 failed 的行走段 |
| 大厅区域首次进入 | sequence 732，XY `(-17.417620, 31.757854)` m |
| 右侧区域首次进入 | sequence 2421，XY `(-16.744452, 32.187210)` m |
| 最终位置 | XY `(-16.733431, 34.252640)` m |
| 最终目标误差 | 0.161699 m，阈值 0.3 m |
| 桌前几何距离 | 1.043360 m，目标为当前可见的办公桌几何 bounds |
| 直立保持 | 158 / 5 控制步；高度 0.116195 m、tilt 0.005546 rad |
| 停止 | 末段 75 个零速度控制步，80 个连续停止样本 |
| 外部障碍接触 | 累计 0 |
| 正式结果 | independent Verifier passed，plan version 8 完成，native tasks.finish，run succeeded |

Planner 在顺时针转向停滞后执行零速度控制、确认停止、读取新观测，并采用左转弧线重新规划。+88°、+288°、+15° 与 +70° 的实际转向完成，误差满足 5°。两次顺时针转向返回 `blocked`。

长距离工具精度保持开放：1.3 / 1.25 / 1.6 m 三段终点误差分别为 0.093094 / 0.078528 / 0.055051 m，均返回 `failed`，具有五个实际停止样本。Planner 使用测得位置完成路线。当前 run 的严格 `--require-metric-tools` 检查拒绝通过，记录为 `Metric walking lacks a completed measured target`；路线完成与距离工具精度分别记录。

## MP4 与检查

视频为 `outputs/demos/oh-my-duck-office-vln-20260930.mp4`：1920×1080、10 fps、1933 帧、193.3 秒。画面包括实际 observer、带目标框的 head capture、物理采样编号、Planner 公开文字、计划、工具参数/反馈、正式 verdict 和最终状态。等待时间按 24× 播放，737 个实际运动相机帧的时间间隔保留原生 simulation time 下限。

完整导出包含 5507 个原始事件、737 个 observer PNG、22 个 head PNG。视频截至 `run.succeeded` 的第 5492 个事件。PNG SHA256、字节数、采样身份、视频时间与每个编码帧的文字可见边界检查通过；FFmpeg 完整解码通过。head RGB 图像的最大通道标准差范围为 33.626552–65.354783，检查未发现全黑图像。

| 检查 | 结果 |
|---|---|
| 原生执行、暂停、停止、正式 verdict 与相机来源 | [通过](evidence/vln-native-20260930.json)，检查 17 次停止进度 |
| 多阶段路线、实际距离、多个工具调用与桌前几何距离 | [通过](evidence/vln-route-20260930.json) |
| 视频来源、时间与文字边界 | [通过](evidence/vln-video-20260930.json) |
| 既有 313 控制步 Newton 距离工具记录 | 严格 metric / 停止 / 正式结果回归通过 |
| 既有 760 控制步 CPU 公寓记录 | 原生执行 / 暂停 / 正式结果回归通过 |
| 当前长行走段的严格终点精度 | 未通过，三段误差保留 |

视频 SHA256：`39fe2c1f07d19f9c5bfc436a44ab5d0e5a0ebe31476f471173ef8cbd9bdae3fa`。

原始事件 SHA256：`de4665910c5078dadfa32cf29a37eb58e415cb3ae77d17365ebbdcbaa9a8060e`。

原始 run SHA256：`fd8805f9c9262b9f5b95f3db3c21405a1a178f32f45b47cb659a771dd6e4e920`。

backend SHA256：`d78166b12b5cfb5505c9b07ff032fdaeb9e434d5a156ecc6be9c13f1d3fff5d1`。

renderer SHA256：`5fa0aece66ee9ea1d85b186c0144db0118680bcf9abebadbb4a81980765865e8`。

## 产物与资源

本地原始导出为 `.cache/perception/replay-03`，服务数据为 `.cache/harness-perception-vln-20260930-03`。实际导航源码副本为服务器 `.job-sources/perception-vln-20260930-03`；YOLO26 服务与诊断使用 `.job-sources/perception-vln-20260930-04`，输出为 `outputs/perception-scene-yolo-04`。失败产物与严格精度拒绝记录保留。

session `d5a5b943-dd0a-4c19-9a79-17222466bcfc` 已关闭，resources 为 released。仿真 worker、模型服务与本地 Harness 服务均已退出；GPU 1/7 为 1 MiB，GPU 5 为 5 MiB，当前没有本项目 GPU worker。RL 保持停止。

SAM 3.1 联合推理、真实图像中的目标识别准确率、长行走段的严格终点精度、图像输入独立 VLN policy、多场景泛化和真机继续需要各自的实际行为验收。
