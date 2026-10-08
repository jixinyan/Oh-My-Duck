# 无 GPU 发布检查 — 2026-10-07

GPU 验收及 RL 训练保持停止。本轮使用本地实际 CPU MuJoCo/BAM、固定官方 ONNX policy 与原生 EDH ActionGate，检查验收进程中断和资源释放。

| 操作 | 实际控制步 | supervisor 退出编号 | 原生会话 | 清理耗时 |
|---|---|---|---|---|
| 控制连接关闭 | 85 | 130 | closed，resources_released | 0.717 秒 |
| SIGTERM | 85 | 130 | closed，resources_released | 0.779 秒 |

输出目录为 `outputs/acceptance/campaign-cancel-eof-20261007-01` 与 `outputs/acceptance/campaign-cancel-sigterm-20261007-01`。每个目录保留原始物理样本、相机、取消结果、会话清理结果和所属进程的退出记录。运动结果保留 failed，清理检查为 passed。没有启动 CUDA 或 GPU worker。

验收控制器和 metric campaign 使用 `OwnedProcess` 管理直接启动的进程。远程 supervisor 通过 SSH 输入连接的关闭接收取消请求，并等待所属 campaign 和物理 worker 清理。超过清理时限会保存终止信号与退出编号，同时返回失败。导航 runner 负责关闭自己的原生会话，控制器随后关闭自己的 server。

Hospital 后退记录的修正运动从约 sequence 237 持续到 332。sequence 327 属于途中命令参数更新，停滞判断仍使用连续运动历史。当前 0.05 米距离要求和 motion guard 保持原有设置。轮滑低速修正响应、长距离和顺时针旋转效果需要实际 GPU 验收。

固定源码 `caac1c9` 的 CPU 连续动作检查及独立复核通过，输出为 `outputs/acceptance/offline-sequence-20261007-01`。五次操作为 +1.0 米、−78°、−0.5 米、+78°、−0.5 米，最终误差依次为 0.021102 米、4.496450°、0.035872 米、4.712026°、0.040380 米。1208 个控制步与 4832 个物理子步，停止、来源、相机、重复观测和资源释放检查通过。

固定源码 `caac1c9` 的 CPU 三组五动作矩阵及独立复核通过，输出为 `outputs/acceptance/offline-matrix-20261007-01`。最大距离误差为 0.035112 米，最大旋转误差为 3.777766°。全部所属进程正常退出、原生会话释放资源。实际 backend 检查覆盖 151 个控制步、站立保持、部分物理子步执行、重放拒绝和丢弃操作，结果保存为 `outputs/acceptance/offline-backend-contract-20261007-01.json`。

十个固定官方 ONNX model 均通过来源 SHA256、ONNX checker、61→14 维度和 CPU 有限输出检查，结果为 `outputs/acceptance/offline-official-policies-20261007-01.json`。Graph 检查不提供对应技能的运动效果结论。十项固定音色资料检查通过，使用实际参考 WAV、SoundFile 与 SQLite，覆盖确认、修改、并发和文件完整性。实际 BAM/ToF 检查四项通过，包含两个附加检查。

发布参数检查共 16 项通过，使用正式六阶段计划和实际场景配置，覆盖重复阶段编号、非有限数值、来源路径、模型、设备、renderer、出生位置和预算类型。`doctor --metadata-only` 与 `run_release_campaign.py --preflight-only` 提供无需 CUDA 的远程检查入口，检查记录与行为验收分别保存。

固定源码 `793883738699ff5733ecf9f42f6df663ad6827d1` 的远程准备检查通过，完整记录为 `outputs/acceptance/offline-preflight-20261007-02`。六个阶段使用四份场景配置、九个来源文件；Office 的 2291 个资源文件及 Hospital 的 1639 个资源文件全部通过大小和 SHA256 检查，场景 USD、公用地图、标准脚和轮滑机器人资源身份通过核验。每份配置的十个官方 policy 均通过检查。远程部署的 29 个原生 Harness Python/schema 文件与固定 Git 内容一致；本地 Node 入口和固定来源同时核验。Provider 配置使用 `gpt-6-astra` 与 Responses API，记录仅保存模型和 API 类型。

准备结果为 `preflight_passed`，`cuda_runtime_checked=false`、`gpu_acceptance_performed=false`。GPU 动作矩阵、Newton 连续导航、识别准确率、物体效果、训练 policy 行为与真机验收仍需相应环境的实际运行证据。

感知客户端核验当前帧的 episode、sequence、时间、查询、图像 SHA256、距离来源、模型身份、目标几何及标注图像尺寸。服务在模型调用前检查 camera/body 位置及 yaw 的有效性。已有实际 Newton RGBD 记录的两张图像、四个 YOLO26 目标通过检查；有效深度数量与目标位置根据原始 world points 独立复算一致，48 项无效资料检查全部拒绝。结果为 `outputs/acceptance/offline-perception-records-20261007-01.json`。该检查使用已有采集记录，没有执行新的模型推理。

复现已有资料检查：

```bash
python scripts/accept_perception_records.py \
  --capture outputs/recorded-rgbd-20260930 \
  --output outputs/acceptance/perception-records-NEW.json
```

输入目录必须包含原始 `result.json`、`head-INDEX.png`、`models-INDEX.png` 和 `points-INDEX.npy`；所有输入 SHA256 保存到结果中。资料缺失或检查失败会立即返回错误。

Policy 调试记录保存实际 61 维输入、13 维命令、14 维输出、官方 model SHA256、执行编号、generation、输入与结果 sequence，以及执行之后的 world velocity。记录在原生 update 序列化前加入；ActionGate、BAM、50 Hz 和物理子步设置保持原有内容。每个 metric campaign case 自动执行独立 ONNX 重算，并保存 `policy-verification.json` 及其 SHA256。

固定源码 `d0ae8f6` 的 CPU 五动作矩阵通过运动复核与 policy 复核，共 1348 个控制步，使用 `velstand` 与 `alpha_walking`；所有重算输出与实际动作误差为零。固定源码 `94db369` 的五动作连续序列通过自动 policy 复核和独立运动复核，共 1208 个控制步，重算误差同样为零。输出为 `outputs/acceptance/offline-policy-input-matrix-20261007-01` 与 `outputs/acceptance/offline-policy-input-sequence-20261007-01`。同一源码的控制连接关闭和 SIGTERM 检查分别执行 84 与 81 个实际控制步，全部会话释放资源，结果为 `outputs/acceptance/offline-policy-cancel-eof-20261007-01` 与 `outputs/acceptance/offline-policy-cancel-sigterm-20261007-01`。

同一源码通过实际 `ground_pick` 终止和 `alpha_stand` 接续检查：技能执行 140 个控制步，接续执行 100 个控制步，确认停止时具有 95 个连续停止样本。转换保留原有 episode、sequence、simulation time 和物理位置，结果为 `outputs/acceptance/offline-policy-transition-20261007-01.json`。该检查覆盖策略转换与站立恢复。

固定源码 `6e3e0afcf3f3fec319c05e3ce53974ec503c108d` 构建 wheel 与源码包，并安装到独立环境；安装包、源码包和实际安装的 390 个源码及资源文件逐个核验一致，三份许可证核验通过。十个 CLI 入口在项目目录之外检查，包括帮助、任务列表、framework 列表、状态、voice-client doctor、语音帮助、voice-task、voice-session、harness 和 sim；实际 voice-client 依赖来自锁定文件，doctor 没有初始化 CUDA。结果为 `outputs/acceptance/offline-installed-package-20261007-01.json`，安装包位于 `outputs/packages/offline-release-20261007-02`。模型命令在调用时直接加载各自依赖。

发布计划检查在项目自己的 `outputs/tests/release-plan` 创建输入资料，支持使用共享 cache 的固定源码目录。当前 16 项检查通过；Python 编译、Node 入口语法和 Git 空白检查通过。远程进程检查没有发现 Oh-My-Duck、Microduck 原生 worker 或相关语音服务运行；已有 GPU 进程的用户身份已核查，当前项目没有分配设备。
