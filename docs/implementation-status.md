# Implementation status

> 2026-10-08：训练 policy 包已经接入原生工具与距离/角度工具的 policy 选择。实际 Walking 与 StandUp 包通过 CPU 执行、限定时长结束、站立策略接续和停止检查：725 次控制、2900 个物理步、145 个原始相机帧、87 个停止样本；独立 ONNX 重算与原始记录检查通过。37 项测试、429 个安装文件、24 个调用及六阶段准备检查通过。四条实际 Newton 构造路径在申请 CUDA 资源之前拒绝无效包；GPU 与 RL 保持停止。学习行为验收仍需推进，见[训练包接入验证](reports/policy-packages-2026-10-08.md)。

> 2026-10-08：公开 `omd validate pose` 与 `pose-audit` 完成原生 CPU 头部及身体姿态验证。575 次控制、2300 个物理步、正反向响应与归零响应、115 个原始相机帧、95 个停止样本及资源释放通过核验；软件渲染身份已经检查。30 项测试、425 个独立安装文件与 22 个调用通过，当前源码六阶段发布准备检查通过。GPU 和 RL 保持停止，见[姿态验证](reports/native-pose-2026-10-08.md)。

> 2026-10-08：官方 solver 接触参数配置位于 `task_binding/contact_model.py`，原生 manager 编译 MuJoCo Warp 数据。标准脚、轮滑及 Office/Hospital 共四个实际 CPU solver model 和原生编译数据通过核验；十项接触参数、独立重算的 body 掩码及 21 项非接触数据检查通过，计算内容的 AST 保持一致。28 项配置及导入测试、423 个独立安装文件与 20 个调用通过，GPU 和 RL 保持停止，见[接触模型验证](reports/contact-model-2026-10-08.md)。

> 2026-10-08：Office 与 Hospital 的实际 CPU 导入、5682 个几何 collider、164 处镜像缩放、接触过滤与两个 finalized model 检查通过；基于 2101169 个原始 points 重算 5680 个导入 mesh 的 world bounds，检查通过，全部来源及生成文件保持检查通过。Hospital 的 125 个 Xform 包含的碰撞几何均已核验。GPU 和 RL 保持停止，见[外部场景 CPU 验证](reports/external-scene-cpu-2026-10-08.md)。

> 2026-10-08：碰撞参数处理位于 `task_binding/collision_model.py`，计算内容的 AST 保持一致。两种模型各两个实际 Newton world 的全部接触过滤、ground pair、摩擦与接触参数通过检查，两个 finalized CPU model 的数组复核通过。27 项配置及导入测试、422 个独立安装文件和 20 个调用通过，GPU 和 RL 保持停止。见[碰撞模型验证](reports/collision-model-2026-10-08.md)。

> 2026-10-08：Newton 的训练、评估、诊断、直接环境创建与原生 RSL worker 共九条实际路径通过依赖异常检查，均在初始化 CUDA、Isaac、pxr 和 W&B 会话之前终止。锁定安装修复后，244 个 provider 文件检查通过。26 项配置及导入测试、源码编译、421 个独立安装文件与 20 个调用通过，GPU 和 RL 保持停止。见[入口验证](reports/newton-execution-preflight-2026-10-08.md)与[CPU 开发验证记录](reports/cpu-development-readiness-2026-10-08.md)。

> 2026-10-08：公开 `omd validate model-assets` 通过两种模型、同一进程的四次实际 Newton CPU 导入、全部 servo 与四个 passive wheel joint、材质和几何保持检查。八条有序过程记录通过独立复核；无效重复次数、重复模型、缺少 CPU 使用限制和已有输出目录均被拒绝，原有输出完整保留。24 项配置及导入测试通过，独立安装核验 421 个文件、三份许可证与 20 个调用。GPU 和 RL 保持停止，见[公开 CPU 资产验收](reports/model-assets-cli-2026-10-08.md)。

> 2026-10-08：Newton 环境使用 `usd-exchange 3.0.0` 提供 OpenUSD 26.08，244 个安装文件通过实际 hash 检查。两种机器人 USD 的碰撞准备、材质绑定与 Newton CPU 导入通过，全部 14 个 servo 保留；229 个 mesh、1285270 个 points、world transforms 和生成资产文件保持一致。碰撞准备位于 `task_binding/collision_assets.py`，Newton 模型 callback 位于 `task_binding/collisions.py`。实际 RSL-RL/SB3 安装、原生 Harness 依赖、六阶段发布准备、23 项配置及导入测试、420 个独立安装文件和 19 个调用通过。GPU 和 RL 保持停止，见[CPU Newton 资产验证](reports/openusd-readiness-2026-10-08.md)。

> 2026-10-08：独立安装检查支持明确的命令时限，每次启动、退出、耗时和输出 SHA256 保存为连续记录。默认完整检查通过 418 个文件、三份许可证、19 个调用及实际 ONNX/物理资料复核；明确时限检查通过 17 个调用。实际子进程中断、保留过程记录、重复输出拒绝和无效时限拒绝通过，见[验证记录](reports/installed-audit-lifecycle-2026-10-08.md)。GPU 和 RL 保持停止。

> 2026-10-08：原生 worker 位于 `integrations/edh/`，分别管理环境、动作设备、会话工具与进程通信；公开 executable 保持 `oh_my_duck.integrations.edh_native`。实际 CPU 连续动作、1208 个 ONNX 重算、任务权限检查、原生 SDK 与独立进程通信、policy 转换及两种运动期间取消全部通过。两种 robot conversion asset 支持经过完整来源检查的明确复用；当前源码六阶段准备检查通过，核验四份场景配置、十个官方 policy 和 29 个 Harness 文件。独立安装检查 418 个文件、三份许可证和 19 个命令调用，远程编译与实际导入通过，GPU 和 RL 保持停止。见[资产验证](reports/source-verified-assets-2026-10-08.md)与[原生 worker 验证](reports/native-worker-modules-2026-10-08.md)。

> 2026-10-08：验收实现按职责位于 `validation/{metric,harness,release}`，原生记录导出位于 `experience/harness_replay.py`。公开入口及 21 项配置与导入检查通过；实际 CPU 连续五动作、1208 个 ONNX action 重算、两种运动期间取消和独立复核通过。独立安装核验 412 个文件、三份许可证和 19 个实际调用；已有 Hospital 导航记录通过完整复核。远程六阶段准备检查通过且没有初始化 CUDA，见[完整记录](reports/validation-modules-2026-10-08.md)。GPU 和 RL 保持停止。

> 2026-10-07：无 GPU 检查通过实际 CPU 五动作矩阵、五动作连续序列、独立复核、原生 backend 行为、十个官方 ONNX graph、十项固定音色资料检查和两种实际运动取消。远程发布准备检查通过六个阶段、四份场景配置、29 个固定 Harness 源文件和全部 Office/Hospital 资源文件的 SHA256 检查，没有初始化 CUDA。GPU 验收及 RL 训练保持停止，见[检查记录](reports/offline-release-validation-2026-10-07.md)。

> 感知资料检查通过两张已有实际 RGBD 图像和四个模型目标，原始 world points 的有效数量及目标位置独立复算一致，48 项无效资料均被拒绝。当前帧来源、时间、模型、目标几何、标注图像及 local endpoint 检查已经接入感知入口。

> 原生 update 保存实际 policy 输入、命令、输出、model SHA256 与执行身份，每个 metric case 自动执行 ONNX 重算。当前 CPU 矩阵及连续序列共 2556 个控制步重算误差为零，实际运动独立复核及两种取消检查通过。

> 独立安装环境核验 390 个源码及资源文件、三份许可证和十个项目目录之外的 CLI 入口，全部通过。实际 `ground_pick` 执行 140 个控制步，`alpha_stand` 接续 100 个控制步，确认停止时具有 95 个连续样本。GPU 验收和 RL 保持停止。

> 2026-10-07：统一观测与原生等待接口通过固定源码的 CPU 连续动作及 Newton Office 五动作矩阵，最大 Office 距离误差为 0.022404 米，最大角度误差为 3.239445°；原始记录通过独立复核。Hospital 第一组通过，后退动作停滞且误差 0.052917 米。GPU 验收已按用户要求停止，本地控制进程及远程 worker 全部退出，剩余阶段尚待验收。后续分配要求设备没有 compute PID，不能与 haomin 或其他用户共用 GPU。见[证据记录](reports/runtime-observation-acceptance-2026-10-07.md)。

> 2026-10-07：当前控制器的 Office 有序路线 run `a8dd0f45-4480-4e60-a89a-0b783f2627c0` 为 failed，最终保持恢复调用超出原有 2400 秒预算，没有正式 Verifier。三段行走全部满足 0.05 米精度，端点位移累计 3.052359 米，最终目标误差 0.116789 米；两次顺时针旋转停滞，模型根据新鲜 ToF 和有界命令继续绕行。2627 控制步 / 10508 物理子步，零外部接触，4266 个事件与 549 张原始图片完整保存，自动入口关闭会话并释放 worker。后续需要验证长行走后的顺时针响应、预算内正式结束和 180 秒初始化可靠性。见[导航测量](reports/navigation-acceptance-2026-10-07.md#office-有序导航测量)。

> 2026-10-07：Hospital 轮滑的有序导航通过真实模型、原生 Harness、Newton/BAM、独立 Verifier 和原始记录复核。三段行走端点位移累计 5.438808 米，1903 控制步 / 7612 物理子步，最终误差 0.095637 米，80 个连续停止样本，累计外部接触为零。完整 MP4 为 170 秒，393 张原始图片与公开 agentic trace 保留；会话和 worker 已释放。两次动作的严格精度未通过，实际结果与后续规划完整保存。自动验收入口、路线参数与证据见[导航验收](reports/navigation-acceptance-2026-10-07.md)。

> 2026-10-07：完整 Office 原生任务通过四个官方 policy 的技能和导航验收：坐立高度变化 0.054849 米，head yaw 变化 0.333259 rad，ground_pick 完成 140 个控制步，直立恢复后连续停止 97 个样本。999 个控制步与 3996 个物理子步，独立 Verifier passed，run succeeded，最终目标误差 0.045512 米，连续停止 80 个样本，外部障碍接触为零。六次当前图像的 SAM3.1 + YOLO26 调用保留空目标结果，距离和导航使用 simulator ground truth。标准脚部署使用官方 sitstand 的 30/50 求解参数与 Newton 直接执行，会话及服务已释放。见[完整测量](reports/office-skills-acceptance-2026-10-07.md)。GPU 同时最多使用一张设备；RL 保持停止。

> 2026-10-07：同一控制器源码在 CPU 公寓、Newton Office、Newton Hospital 完成九个独立会话和十五次运动，每个场景覆盖 +0.5 米、−0.5 米、+1.0 米与 ±45°。CPU 最大距离/角度误差为 0.035112 米/3.777766°，Office 为 0.021602 米/4.952898°，Hospital 为 0.049210 米/3.746165°。全部动作确认直立停止、零外部障碍接触与会话释放；连续物理样本、相机字节、执行计数和来源在本地独立复核通过。见[完整测量](reports/metric-controller-acceptance-2026-10-07.md)。

> 物理状态已经停止且原始请求误差满足要求时，metric controller 进入零命令制动，并重新检查最终误差和五个停止样本。旋转根据实际制动角度进行最多三次修正；50 个控制步内目标进度不足时返回 `metric_progress_stalled` 并确认停止。ONNX Runtime 在初始化前关闭遥测，三个 CPU 仿真子进程正常退出。GPU 检查顺序使用 GPU 4，相关进程已退出；授权范围为 2–4，同时最多使用一张设备。RL 保持停止。

> 2026-10-03：录音指令经 Qwen ASR、原生 Harness、真实模型、官方 policy、Newton/BAM 与独立 Verifier 完成任务，并生成固定音色反馈。Office 任务 succeeded，目标误差 0.090916 米，480 控制步与 1920 物理子步；91.4 秒 MP4 包含实际画面、公开 agentic trace 和语音。标准脚 0.5 米工具误差 0.024196 米，45°工具误差 0.522914°；轮滑模型对应误差为 0.018337 米和 3.893902°，四项均取得五个实际停止样本。安装包、项目目录之外的语音入口、实际服务检查和十项音色资料测试通过。CPU MuJoCo/BAM 与 Newton 的执行期间中断通过，停止确认之后两秒内动作计数保持不变，会话释放资源。Office 新进程初始化符合原生 Harness 的 180 秒时限，使用已有 kernel 缓存。其他命令设置、多场景泛化、识别准确率、物体效果、训练 policy 行为和真机需要独立证据。见[上线要求](release-readiness.md)与[完整证据](reports/end-to-end-2026-10-03.md)。

> 2026-10-01 多 policy Demo：NVIDIA Hospital 的官方 roller 模型、四个 passive wheel joint、`roller` 与 `crouch` 已通过真实模型、原生 Harness、Newton/BAM 和独立 Verifier 的目标验收。677 个控制步、2708 个物理子步，最终误差 0.1884 米，直立保持 42 个采样周期，26 个连续停止样本，外部障碍接触累计为零；三个当前图像的 SAM3.1 + YOLO26 调用返回六个有效目标，距离使用 simulator ground truth。63.4 秒 MP4、原始图像、事件和物理边界检查通过。请求前进 0.5 米的实际平移为 1.147 米，严格终点精度未通过。Office 的 sitstand、头部控制和 ground_pick 已有独立物理检查；其多动作导航任务因模型请求失败尚未完成。当前会话已关闭，RL 保持停止。识别准确率、RTX 渲染、携带物体、多场景泛化与真机需要独立验收。见[验证记录](reports/multiskill-demos-2026-10-01.md)。

> 2026-09-30 多阶段感知导航：真实 Astra/high 在原生 EDH、Newton/BAM 和官方 policy 上完成大厅中心、右侧绕行与办公桌接近，执行 3689 个控制步/14756 个物理子步，独立 Verifier passed、run succeeded。7 次不同物理帧的桌子观测与目标距离明确标记 simulator ground truth；行走段累计测量 4.1665 米，最终目标误差 0.1617 米，桌前几何距离 1.0434 米，80 个连续停止样本，外部障碍接触为零。3 段长行走的严格终点误差未满足 0.05 米，保留 failed；Planner 根据测得位置和停止后观测完成路线。737 个 observer、22 个 head 图像与 5507 个事件生成 193.3 秒 MP4，原生边界、路线、图像 SHA256、视频时间/文字和完整解码检查通过。会话及 GPU worker 已释放，RL 保持停止。当前为单一 Office seed 的 GT 辅助导航，SAM 3.1、识别准确率、严格长行走精度、多场景泛化与真机尚待验收。见[完整验证记录](reports/perception-vln-demo-2026-09-30.md)。

> 2026-09-30 感知工具：`microduck.inspect_scene(prompt, source)` 已接入原生 Harness，返回当前物理帧的目标框、表面距离、水平距离、bearing 与标注图像。实际 Newton Office 的两个相机朝向得到 21 个有效 ground truth 目标；独立 SAM3.1 + YOLO26 服务使用 jd_B300 已有权重，完成实际 RGBD 的墙面、桌子、地面与植物分割和射线距离传输检查。空目标与 YOLO 关联类别保留原始结果。识别准确率和真机尚待独立评估。来源与范围见[感知工具](perception-navigation.md)。

> 2026-09-30 距离/角度工具与相机：`microduck.walk(distance_m)`、`microduck.rotate(angle_deg)` 已经通过实际 Newton/BAM、官方 `alpha_walking` 与原生 ActionGate 检查，覆盖 0.4/1.0 米、+45°/−45°/+270°，停止后误差满足 0.05 米/5 度且具有五个实际停止样本。转向返回实际平移距离。head RGB 的官方相机朝向与 0.01 米近裁剪、observer 的场景取景已完成实际 RGB/segmentation 检查；head 包含 46 个场景几何形状，observer 包含 12 个场景形状与 5114 个机器人像素。真实 Astra/high 在 Newton Office 调用距离工具，执行 313 控制步/1252 物理子步，独立 Verifier passed、run succeeded，正式目标误差 0.0585 米；782 个事件、62 个 observer 帧和两个 head 观测生成 60.3 秒 MP4。原始 metric motion、实际命令与停止进度检查、视频来源/时间/文字边界及完整解码通过；GPU worker 已释放，RL 保持停止。当前行为范围为单一 Office seed，CPU 距离/角度、多场景长导航、物体效果与真机尚待验证。见[验证记录](reports/metric-camera-tools-2026-09-30.md)和[工具接口](metric-policy-tools.md)。

> 2026-09-30 Isaac agentic Demo：真实 Astra/high 通过固定 EDH 原生运行时与 SSH GPU worker，在 NVIDIA Office 的 Newton/BAM 环境执行官方 `velstand` 与 `alpha_walking`。累计 300 个控制步、1200 个物理子步，实际位移 0.4047 m，外部障碍接触累计为零；末段执行 100 个零命令控制步，确认 78 个连续停止样本。独立 Verifier 判定 passed，目标误差 0.0988 m、直立目标保持 114/5 个控制步，Planner 完成计划并调用原生 `tasks.finish`，run 为 succeeded。完整记录包含 899 个事件、60 个实际 observer 帧与两个头部相机观测；MP4 显示公开 Planner 文字、计划、工具反馈与正式 verdict。原始记录与停止样本检查通过，GPU 会话已释放，RL 保持停止。量程内原有墙面检测与 89 mm 近障碍暂停另有独立验证。当前继续事项为 RGB 地标可辨认度、长距离导航、多场景与多 policy 物体效果。见[agentic Demo](reports/isaac-agentic-office-demo-2026-09-30.md)、[近障碍停止与可视化](reports/isaac-proximity-2026-09-30.md)与[场景资源](reports/isaac-scene-assets.md)。

> 当前核查（2026-09-30）：用户已停止 Walking 低速干预的全部八组配对训练及预览、评估进程；不自动恢复。固定训练源码为 `b8c36b2`，每组原定 8192 环境、50000 次更新，完整预算与最终行为验收均未完成。已有 checkpoint、输出与 W&B 文件保留，详见[停止记录](reports/rl-walking-user-stop-2026-09-30.md)。此前 MuJoCo/RSL-RL Walking 与 Newton/RSL-RL StandUp seed 42、43 的三组完整训练、导出和封装均已完成，最终结果均为 `behavior_failed`，详见[上一轮项目结果](reports/project-status-2026-09-29.md)。语音 HTTP 服务、Mac 内置扬声器播放、内置麦克风录制与合成期间停止后的无迟到播放已通过实际验证；合成原文为「你好，我是小鸭。我们现在检查语音连接。」、麦克风回读为「您好，我是小丫。我们现在检查语音连接。」，详见[语音交互验收](reports/voice-interaction-validation-2026-09-29.md)。原生 EDH 公寓会话已完成真实图像、工具、ActionGate 动作、`policy_stop` 和独立 Verifier 闭环；固定出生位置的独立 office 导航获得正式 passed，全程外部障碍接触累计 0，详见[公寓导航验收](reports/harness-office-navigation-2026-09-30.md)。项目目前没有可用真机。

> 2026-09-30 Walking 因果干预门禁完成时记录：针对保存策略在 `0.1 m/s` 近似静止、`0.2–0.4 m/s` 才响应的证据，新增的 `low_speed_tracking_boost` 配对实验完成 8 个 run 的 smoke、导出、恢复、8192 容量和 CPU/BAM rehearsal 门禁；默认值为 0，官方 Walking 配方不变。当时按最小目标在门禁后停止，尚未进入完整训练，未产生可封装 tool 的策略。首次门禁暴露并修复了 managed-host EGL loader 传播问题，Newton 资产按当前 fingerprint 重建后四组门禁通过。后续启动记录见[Walking 启动记录](reports/rl-walking-low-speed-launch-2026-09-29.md)，目前状态见[停止记录](reports/rl-walking-user-stop-2026-09-30.md)；门禁证据见[低速干预](reports/rl-walking-low-speed-intervention-2026-09-30.md)与[门禁记录](reports/rl-walking-low-speed-gates-2026-09-30.md)。

> 2026-09-30 ProtectiveFall 任务入口：已把完整碰撞资产、`servo_impact_contact`、伺服 stall/加速度保护项和 fallen smoothness scaling 组合为独立 `Mjlab-ProtectiveFall-Flat-MicroDuck` 配置，CLI 标为 `experimental_unvalidated`。官方 VelStand、Flat Walking 和 Flat StandUp 配置未被改写；该入口只通过静态配置/14-servo 编译测试，尚未训练、Newton 验证、导出、CPU/BAM 演练或封装为 tool。

> 2026-09-30 官方 develop/main 迁移续项：已把官方 API-2 显式状态 LSTM 的 `float32`/动态 batch/状态形状检查、失败清空状态和私有 reset 边界迁入发布校验；保留前馈 API-1。新增 deterministic BT.601/UYVY 相机、8×8 ToF 逐射线保护、官方 v15/alpha4/alpha16 足底 odometry anchor 数据，以及 protective-fall 的 servo stall/acceleration 与 fallen smoothness MDP term。对应 CPU 测试已通过；TCP body server、真实硬件 provisioning、完整 VelStand expert-BC 与 Newton GPU 行为门禁仍未完成，不能把这些接口当作已验证 locomotion tool。

> 2026-09-30 Harness 状态：`omd harness` 使用固定 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd` 原生运行时及独立 CPU 环境。真实 Astra 会话已接收 RGB 图像、ToF、IMU、关节与里程计，调用 MicroDuck 工具；官方 ONNX 策略的 14 关节动作经过 ActionGate，每次执行四个 0.005 秒 MuJoCo/BAM 子步。每条运动命令限定为 5–100 个实际控制步；真实 ToF、外部接触和位姿停滞可触发原生 Gate 暂停。独立新会话从 corridor 固定出生位置导航至 office，累计 760 个控制步与 3040 个物理子步，其中最后 100 个控制步使用零命令；确认 87 个连续停止样本，非地面外部接触累计 0。独立 Verifier 判定 `goal_reached=true`、连续达标 108/5 个控制步，Planner 调用原生 `tasks.finish`，run 终态 succeeded。取消、旧 lease 和关闭任务时在途请求的真实边界测试见[真实闭环记录](reports/harness-apartment-live-2026-09-30.md)；本次导航证据见[公寓导航验收](reports/harness-office-navigation-2026-09-30.md)。

> 2026-09-30 官方上游复核：已抓取 `pollen-robotics/microduck_rl` `develop@cfe1c2a` 与 `pollen-robotics/microduck` `main@f0d934e`，并更新 `configs/upstream.json`。已迁移通用的粗糙地形 reset 原点修复，避免把局部高度写成世界高度导致机器人出生在地形内部；新增 CPU 回归覆盖非零 terrain origin。上游 protective-fall/VelStand 的其余语义变化保留为独立后续变体，不改变代表性 Walking/StandUp 基线。详见 [官方上游复核](reports/upstream-review-2026-09-30.md)。

> 2026-09-30 自训练策略状态：jd_B300 的 MuJoCo/RSL Walking 50000 次更新策略在无推扰前进命令 `0.1 m/s` 下仅约 `0.3%` 响应；Newton RSL StandUp 两个 15000 次更新策略在俯卧和仰卧恢复上失败。训练、导出、回放和视频产物保留；这些自训练策略尚未作为已验收 locomotion tool 注册。当前 Harness 公寓接入使用有独立 manifest 来源的官方预训练策略，导航目标继续按原生 Verifier 验收。详见 [jd_B300 诊断](reports/rl-jd-interrupted-2026-09-26.md)。

> 2026-09-26 RL 状态：`jd_B300` 上的三组正式训练在完成预算前同时中断，原始 checkpoint 和日志均已保存；退出代码 247 的原因尚未确认。Walking SB3 保存于第 6000 次更新，原生 MuJoCo 无推扰评估完成 700 步，但前进与转向响应未通过评分；该策略目前保留。Newton RSL StandUp 两个 checkpoint 各已完成 5001 次真实更新，原生 Newton 无推扰评估在单一 seed 下均为 2/4 姿态。新的 MuJoCo/RSL Walking 官方配置对照已在 GPU6 进行完整训练；原生恢复计数修复通过真实 checkpoint 连续恢复验证，两组 Newton StandUp 已在 GPU2/GPU1 接续剩余 9999 次更新，训练日志均连续记录第 5001 至 5005 次更新。独立预览流程用真实 StandUp checkpoint 生成四份视频。已检查的 TensorBoard 标量均为有限值，现有训练奖励不能代替行为验收。详见 [中断训练诊断](reports/rl-jd-interrupted-2026-09-26.md) 与 [官方实现核对](reports/rl-official-comparison-2026-09-26.md)。

> 2026-09-23：`jd_B300` 的 CUDA 13 正式环境通过实际 GPU 检查，macOS 和 Linux 分别通过同组 55 项 CPU 测试。Isaac 资产转换通过，三组训练均完成启动检查、导出、恢复与容量检查，并已进入完整预算：Walking 使用 GPU 6，Newton StandUp 的 seed 42、43 分别使用 GPU 2、1。三组在线 W&B 记录已开始并出现正常更新；短训练回放行为未达标（返回码 2），最终行为验收需要完整训练与评估。语音与仿真执行仍待实现。GitHub 开放 Issues 和 Pull Requests 均为 0。见 [当前状态与未完成工作](reports/project-status-2026-09-23.md)。下方保留历史验证记录。

> 2026-09-21 CPU 验证：Walking 使用逐阶段评分版本 3；训练前检查复用覆盖完整源码、配置及依赖声明；SB3 按指定目录恢复并核对训练预算。53 项 CPU 测试通过，包括实际 Git 操作和原生 PPO 保存、加载、继续训练。Microduck GPU 仿真和历史策略重新评分尚未执行。见 [验证记录](reports/project-review-2026-09-21.md)。

> 2026-09-21 服务器迁移：按用户最终决定，仅迁移 Git 管理的源码、配置、锁文件与文档，合入并推送 main；checkpoint、normalizer、日志、视频、离线 W&B 和本地环境不上传。完整数据归档已取消，原训练产物保留在旧文件系统。原 Walking job 已不在调度 API 中，不自动恢复；既有策略结论保留为历史证据，不代表重新训练必然复现。见 [迁移交接与恢复说明](server-migration-2026-09-21.md)。

> 2026-09-14 策略复盘：231 份已完成周期预览均未通过完整行走标准。最近五个 checkpoint 中，MuJoCo SB3 官方课程、MuJoCo RSL 延后课程相对更值得做无推扰验证；Newton RSL 延后课程后期前进退步，Newton SB3 两组前进仍弱。延后平滑没有通用收益。横向瞬时误差包含快速摆动，不能直接等同持续侧滑；暂停状态和验收标准不变，本次仅分析既有视频与轨迹。见 [策略表现复盘](reports/rl-walking-policy-assessment-2026-09-14.md)。

> 2026-09-14 暂停状态：调度器将 `omd-walk-pacing-0913-01` 标记为 **Suspended**，八组日志均停止在 07:38 UTC 左右，本地 `running` 为滞后记录。暂停原因未返回，不能归因于代码或调度抢占。各组最后日志约 19803–41043 / 50000 更新；最近 checkpoint、normalizer、日志及视频保留，最终验收未执行。本次查询未重启或重新提交任务。见 [暂停与 checkpoint 记录](reports/rl-walking-suspension-2026-09-14.md)。

> 2026-09-13 完整训练迭代：前轮因果 job 已 Succeeded，32 个回放及两个 Newton 任务的全部门槛完成，四份周期导出与官方路径数值误差均为 0。无推扰对照确认 SB3 主动行走很弱；Newton RSL 在两后端的 play 配置下均有前进/转向响应，训练配置差异仍待定位。下一轮已提交单节点 8 GPU 的 Walking 配对完整训练（`omd-walk-pacing-0913-01`，已 Running，八组流水线已启动）：四种后端/框架组合，各比较官方任务课程和延后动作平滑课程，8192 环境、50000 更新；同一 job 内检查通过自动训练，定期视频与最终无推扰/跨后端/CPU 验收。该课程调整是待检验假设，不是已证实修复；StandUp 后续验收仍开放。见 [完整实验记录](reports/rl-walking-pacing-2026-09-13.md)。

> 2026-09-13 调度方式更新：后续常规 job 一次执行必要启动检查、完整训练预算及最终评估/视频；检查通过后自动继续，不再常规单独提交短验证 job。当前已在运行的 `omd-rl-causal-0913-01` 仍是原定短验证任务：32 个回放已完成，两任务周期导出已通过数值一致性检查，StandUp 最后容量检查尚在执行。

> 2026-09-13 诊断完成：`omd-rl-diagnose-0912-02` 的 49 个用例全部执行，不能等同于行为通过。Newton RSL 最终起身在两后端 seed=42 均 4/4；CPU/BAM 17 个种子中趴倒 14/17，其余三种姿态各 17/17。其他起身策略仍为 2/4，Walking 未完成验收；Newton Walking 有前进/转向能力，但横向摆动与跨后端前进失败需分开诊断。下一轮补齐只改变推扰强度的配对对照和视频，并验证 Newton 周期导出的真实训练回调；暂不盲目恢复完整训练。下一轮 `omd-rl-causal-0913-01` 已 Running（1 节点 4 张 H800，32 个回放对照后执行两任务短训练门槛）；24 项 CPU 测试与 9 个子测试通过。详见 [因果回放记录](reports/rl-causal-replay-2026-09-13.md)。以下为历史快照。

Updated 2026-09-12. RL iteration resumed by user instruction; single-node,
multi-GPU scheduler jobs are authorized. A 49-case frozen-policy diagnostic
matrix is prepared, including weighted reward traces and 17-seed CPU/BAM checks
of the final Newton StandUp policy. Diagnostic training-stage/push interventions
are explicitly ineligible for standard acceptance. Ten protocol/profile tests and three native export-metadata/callback tests
pass. Newton periodic export metadata indexing is repaired; GPU callback
integration remains pending. Diagnostic job `omd-rl-diagnose-0912-02` is accepted
(1 node, 4 GPUs) and waiting for project quota; attempt 01 failed on a shared lock.
See [current diagnosis](reports/rl-learning-diagnostics-2026-09-12.md). New tasks remain inventory entries
until their own validation, evaluation and packaging gates are implemented.
See [task catalog](rl-task-catalog.md). The pause below is historical.

Updated 2026-09-09. Product scope remains the Agentic Microduck Project Design.

**Training paused by user; zero project RL processes remain.** Five remaining
learners and both live preview workers were stopped with preserved checkpoints,
logs, videos and identity/hash records. No automatic resume is scheduled.
Newton RSL StandUp final passes 4/4 at seed 42 in Newton, MuJoCo and CPU/BAM;
final multi-seed robustness is open. MuJoCo RSL StandUp seed 43 completed with
2/4. Repaired SB3 has stable updates but incomplete learned behavior; Walking
also remains below acceptance. Next priority is causal debugging, not additional
long training. See [current state and debugging handoff](reports/rl-debug-handoff-2026-09-09.md)
for exact saved steps, evidence and the investigation sequence. CLI maturity
reflects partial behavior with training paused. All notes below are historical.

Latest operational update (2026-09-09): four old Walking learners were
intentionally stopped after repeated behavior failures: owned MuJoCo RSL, old
MuJoCo SB3, old Newton SB3, and the original 4096-env MuJoCo RSL control. All
artifacts and checkpoint hashes are preserved with intentional-stop records.
Six learners continue: Newton RSL Walking, the four repaired SB3 runs and the
MuJoCo RSL StandUp seed-43 control (now in full training). Newton RSL StandUp
finished 15000 updates and passed its final native 4/4 pose preview; final
transfer acceptance remains open. The paragraphs below are earlier snapshots.

Latest repair: native SB3 now has separate official critic observations (74D
StandUp / 76D Walking), serialized KL learning-rate feedback, randomized initial
episode phases and matching terminal/normalizer/export/resume paths. 29 tests
and 9 subtests pass. A paired 32-update continuation reduces mean KL from 0.073
to 0.012; this does not prove learned behavior. All four fresh 8192-env runs passed their individual gates and produced full PPO
updates in `sb3-repair-0909-01`, source `bea3eb1`; offline GPU processes verified.
Newton StandUp 13000 passes 12/12 scenarios in each native backend, but CPU/BAM
prone recovery passes only 3/17. No-push Walking 14000 retains forward/turn
response in both native backends; deployment forward motion fails. Details:
[repair and transfer](reports/sb3-critic-transfer-2026-09-09.md).
The SB3 repair milestone is merged and pushed to main `49b56de`; follow-up is
on `feat/rl-transfer-validation`. The complete 14-group frozen-policy battery
confirms MuJoCo StandUp final remains 9/12 in each backend (all supine failures).
Standard Walking with pushes fails acceptance in both backends; no-push results
are diagnostic only. The seed-43 MuJoCo RSL control (`1cb7738`, GPU 1) is running
fresh startup gates before its unchanged 8192-env/15000-update official recipe.
It has a background checkpoint preview worker starting at update 1000.
The following review and operational notes are earlier snapshots.

Latest behavior review (2026-09-09 02:29 UTC): five of the new 8192-env learners
continue, two MuJoCo StandUp runs completed (RSL 3/4, SB3 0/4), and Newton SB3
StandUp was intentionally stopped after ten consecutive 0/4 previews and
persistent excessive KL. Its update-10000 bundle is preserved and hash-verified.
Newton RSL StandUp repeatedly passes its native four-pose battery; final
sim2sim/CPU and multi-seed acceptance remain open. Newton RSL Walking responds
to forward/yaw commands but misses the lateral RMSE gate. Current SB3 critic
input is actor-only 61D versus official RSL's separate 74D input, alongside
native PPO/normalizer differences. Learner equivalence has not been established.
See [behavior and framework review](reports/rl-framework-status-2026-09-09.md).
The following September 8 operational paragraphs are historical snapshots.

Unused-process cleanup completed: all 26 processes in the old paused
`shared-gpu7-0908-01` tree exited, releasing 23.2 GiB on GPU 7. Current eight
learners, original controls and preview workers remain active; retained checkpoint
hashes are unchanged. See the fixed-size training report for the cleanup audit.

Latest user decision supersedes environment sweeps: all eight new combinations
use 8192 environments. Scaling and the old handover were explicitly stopped;
completed matching gates will be reused, with missing gates run independently.
`configs/experiments/representative-8192.json` keeps native PPO and task budgets,
SB3 learning rate 1e-4, and offline W&B. All eight full learners have produced PPO updates and passed startup checks;
each uses 8192 environments. Live process checks confirm offline W&B. Campaign `fixed-8192-0908-01`, source `3070b71`.
See [fixed-size launch evidence](reports/rl-fixed-8192-2026-09-08.md).

Latest scope: train both tasks across both owned backends and both native PPO
frameworks. `measured-env-0908-03` (source `f9fcf80`) is preparing on GPUs
1/6/2/3/4/5: six initial 64-env/5-update training smokes passed; Newton SB3
Walking/StandUp are queued for the next free card. All eight environment counts
remain to be selected from measured PPO throughput with 15% VRAM headroom.
Full training now starts per combination after its own gates pass. The old
global-barrier supervisor is paused while its live preparations continue.
`independent-full-0908-01` (source `392776a`) now adopts each completed
preparation independently; at startup, all new full learners still awaited their
own remaining gates. New campaigns release GPU slots during CPU stages. Original controls continue on 0/7. Their preview controller is temporarily
paused and will automatically resume after isolated calibration. Prior failed
startup artifacts are preserved. See [selection and startup evidence](reports/rl-environment-selection-2026-09-08.md).

Latest user preference: no continuous LLM training polling. A background worker
saves native videos every 1000 PPO updates (24000 control steps per environment)
and at the final checkpoint; unified CPU video/16-reset diagnosis runs after both
original learners end. Earlier intermediate milestone and paused-owned preview
watchers were replaced. Gallery: `outputs/previews/official-periodic-0908-01/index.html`.

Earlier official-first phase: original MuJoCo Walking/StandUp continue. The user
reopened idle GPUs: Walking remains on GPU 7; StandUp migrates from checkpoint
2000 to GPU 0 through native resume; configuration/curriculum audit passed.
GPU 0 later became shared with a foreign process. Diagnostics use GPU 1; a separate
Walking throughput benchmark measured 4096/8192/16384 on GPU 3, with 8192 best
at 98k samples/s. A foreign eight-GPU job confounded the 32768 case and stopped
the sweep before StandUp. Long learners retain 4096. Original StandUp sitting
success remains 15/16 at checkpoint 2500; prone/supine remain 0/16. Eight original
checkpoint-2000 CPU/native StandUp videos are checked. See [resource/progress review](reports/official-training-resource-review-2026-09-08.md).
Owned default Walking (3299) and StandUp (2985) are paused for diagnosis; three
Newton learners remain suspended and three earlier SB3 attempts stopped. Walking
3000 has only 0.98% forward response when native pushes are zeroed, despite 36.4%
in its standard preview; it also nearly stops in nominal native and CPU replay.
The official-guided StandUp angular-penalty diagnostic completed 2500: paired
sitting improves from default 10/16 to 15/16, but prone/supine remain 0/16 each.
All eight final CPU/native videos and local packaging are checked; own-policy
behavior reproduction remains open. The user reaffirmed official-recipe-first
reproduction; the prepared second tuning experiment is deferred and was not launched.

Official history audit confirms six historical/current train/play configurations
differ only in Walking logging names. Published policy bytes are traced to official
runtime commits; their exact training run remains unknown. At checkpoint 1500,
both original/owned Walking nearly stand still in CPU replay; neither StandUp
recovers from prone/supine. A replacement observer fixes resumed checkpoint lookup
and advances original task assessments independently while owned runs are paused.
See [source-history evidence](reports/official-source-history-2026-09-08.md).

The corrected CPU/BAM matrix now completes all eight preserved acceptance policies
with finite 61/14 traces and 20 checked videos; all short policies still fail
behavior. Native previews use scoring v2 (`outputs/previews/shared-gpu7-0908-02`).
See [baseline audit](reports/official-baseline-audit-2026-09-08.md) and
[StandUp diagnosis](reports/standup-recovery-diagnosis-2026-09-08.md), and
[Walking condition/transfer diagnosis](reports/walking-transfer-diagnosis-2026-09-08.md).
SB3 campaign learning-rate routing is tested; a four-combination 1e-4 configuration
is prepared but not launched. Default task recipes and native PPO algorithms remain unchanged.

| Component | Current state |
|---|---|
| Source organization | First-party code consolidated under `src/oh_my_duck`; obsolete `training/` packages removed |
| CLI/application | Public `validate` / `replay` / CPU assets and pose audits / registered policy packages, Newton execution preflight, native voice workflow, 429 installed files and 24 CLI calls verified |
| Core contracts/tool catalog/recording | 已验证 schema、工具登记与 JSONL；原生 Harness 完整事件和 PNG 导出、SHA256 检查及 1080p agentic MP4 已完成实际验证 |
| Agentic Harness, skills, robot execution | 固定 EDH 原生会话、SSH GPU worker、真实模型、官方 ONNX、ActionGate 与独立 Verifier 已运行；CPU 公寓与 Newton Office 的固定场景导航、Office 四个 policy 的完整多动作任务和 Hospital 轮滑有序路线通过。三个场景各通过三个距离/角度会话；全部长距离请求精度、图像输入独立 VLN、多场景泛化与真机待验收 |
| Perception and policy adapters | 官方十项 policy 与登记的训练包共享 61-to-14 推断及原生工具；实际 CPU Walking/StandUp 包执行、episodic 时长/接续与停止通过。Newton Office 的 velstand/alpha_walking、head RGB、observer 场景、ToF、IMU 与 odometry 已验证；学习行为、物体识别及物体效果待验收 |
| Voice interaction | Qwen ASR、已确认音色 TTS、原生 Harness、Newton/BAM 与独立 Verifier 的录音任务闭环已经通过；Mac 音频设备、合成中断与 CPU MuJoCo/BAM、Newton 任务执行中断通过，会话释放资源。Microduck 音频设备待验收。见[闭环记录](reports/end-to-end-2026-10-03.md)与[设备检查](reports/voice-interaction-validation-2026-09-29.md) |
| MuJoCo RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Isaac/Newton RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Newton physics | Actual solver, canonical state/sensors, BAM cadence, DR and penalties audited; current CPU import/contact checks cover both robot variants and all Office/Hospital geometric colliders; current GPU runtime acceptance pending |
| Published StandUp reference | Official frozen policy passes 64/64 CPU reset samples and 4/4 cases in each native backend; 12 videos checked; own training reproduction remains open |
| Task behavior | All five-iteration policies fail behavior gates; long training and convergence acceptance remain |
| Task replay | All 24 replay contexts completed; 60 videos and finite 61/14 traces checked |
| CPU rehearsal/sim2sim/local packages | All eight corrected CPU/BAM executions revalidated; 20 videos checked; learned behavior remains unverified |
| Multi-GPU | Native RSL MuJoCo DDP previously passed; Newton DDP acceptance pending; old job `a52ff51b` no longer exists in the platform API; no distributed SB3 gradient claim |
| Hardware | Unavailable; all hardware acceptance deferred |

The consolidated batch at `5971dca` passed training and resume for all eight
combinations. Four SB3 exports and local packages passed there. After the merge,
all four RSL checkpoints passed export, numerical parity, finite scalar/penalty
checks and local schema-2 packaging using source `6303ca4`. Evidence:
`outputs/postmerge-rsl-export-0906-01/result.json`. The earlier failures remain
preserved; ordinary parity uses elementwise tolerance and extreme stress inputs
use a reported per-action-vector infinity norm tolerance.

The complete replay matrix used `a0b1f0f`; all eight policies completed both
simulation backends and CPU/BAM rehearsal without runtime errors. Behavior failed
as expected for short smoke checkpoints. Explicit MuJoCo OSMesa video and native
Newton video produced 60 verified 720p clips. SB3 curriculum restoration was then
validated at `b65e8f9` on both tasks/backends, including a second saved-state resume.

Current configuration/import tests passed 28; earlier task/SB3 checks passed 72 cases. Earlier installed-runtime
checks passed seven Isaac and two Newton binding tests. Wheel resources/licenses,
bytecode exclusion, three environment locks and local Markdown links were checked.
See [complete acceptance evidence](reports/rl-pipeline-acceptance.md).

Single-GPU work runs directly on the development host. Following the scheduled
campaign failure, the user also authorized the eight-GPU campaign on the local
H200 host on 2026-09-08; multi-GPU scheduler submission remains available. W&B remains offline. Current host workloads contend for GPUs,
so observed throughput is not an isolated hardware benchmark.

See [domain refactor evidence](reports/domain-refactor.md),
[RL acceptance](rl-reproduction.md), [architecture](architecture.md), and
[historical implementation log](reports/implementation-history.md).

See [headless rendering evidence](reports/rendering-validation.md) for the explicit software renderer and preserved failed attempts.


## Task organization and full-training preparation — 2026-09-07

Task recipes now live in 14 family directories with separate `environment.py` and
`ppo.py`; flat/rough/backlash variants keep the existing registry. All 33 compiled
task catalog entries match the prior semantic inventory (module paths excluded).
Thirty lightweight and 74 task/SB3 tests passed, including native periodic
checkpoint reload and GPU/rank isolation. Full campaigns have a 64-env smoke,
resume and 4096-env capacity gate before the official full iteration budgets.
Long-training submission and worker gates are recorded separately below.


The new campaign worker completed two reduced-size end-to-end gates (RSL-RL and
SB3 Walking) in `outputs/campaign-gate-0907-01`. Every stage completed, including
periodic SB3 bundle resume, final package and both-backend/CPU video. Both reported
`behavior_failed`, with no execution error. These orchestration gates used 64
environments and a two-iteration final segment; the real 4096-environment capacity
gate runs on the allocated training node before long training. Newton binding
checks also passed (two tests). Source/config provenance now includes nested
experiment manifests. Full submission follows from the committed snapshot.


The scheduled attempt `omd-rl-full-0907-01` (platform ID `b2bab260`, original
queue ID `local-662cf40c7dfb`) is **Failed**. Its log is empty, the platform exposes
no log pods, and no campaign manifest exists; the root cause remains unknown.
A separate local attempt `full-local-0908-01` started on eight H200 GPUs using
the same immutable source `1f45996` and training budgets. All eight smoke stages
completed; subsequent gates and full training remain in progress. Failed scheduler
evidence is preserved; local launch metadata is in
`outputs/jobs/omd-rl-local-0908-01/launch.json`.
See [the allocation and evidence record](reports/full-training-2026-09-07.md).


At 2026-09-08 02:08 UTC all eight local runs had entered full training. Early
returns improved relative to initial records; SB3 StandUp shows substantial
regression from its early peak. No learned behavior acceptance is claimed.
See [the measured reward snapshot](reports/reward-trends-2026-09-08.md).


## Triage and one-GPU continuation — 2026-09-08

The two regressing MuJoCo SB3 runs are stopped with artifacts preserved. Six
retained runs resumed native checkpoints on GPU 7 at source `b0fa5fe`; the other
GPUs are released from this campaign. Checkpoint video previews are available at
`outputs/previews/shared-gpu7-0908-01/index.html`. Early RSL task videos show
partial skill progress, with no complete behavior acceptance. A controlled native
SB3 update comparison demonstrates substantially lower KL with smaller learning
rates; stable long-training convergence is still unvalidated. See the
[recovery, diagnosis and video report](reports/rl-recovery-2026-09-08.md).


At the 2026-09-08 04:04 UTC status review, Newton SB3 StandUp also showed
return decline (last 50 logged records 11.95 versus 14.45 previously), elevated
KL (recent values 0.13–0.29) and failure of all four spawn checks on its preceding
preview. It was paused for diagnosis under the user’s standing instruction. Its
latest complete checkpoint at cumulative iteration 1500 is preserved with an
operator-stop record. Five runs continue on GPU 7; this is a triage decision,
not proof that the stopped configuration could never converge.


Official MuJoCo/RSL learned-behavior reproduction remains **unverified**. The
2026-09-08 static audit aligns task/PPO configuration, 228/229 MDP definitions
modulo local imports, BAM and core dependency pins, and the subsequent real-runtime audit verifies the Entity-based StandUp reset
state writes in 15 cases. Matched original/refactor learned-behavior comparison
remains outstanding. See [baseline audit](reports/official-baseline-audit-2026-09-08.md).


Current goal: reproduce official MuJoCo/native RSL learned behavior first, then
match it in Newton and SB3. Isolated pinned-original StandUp PPO smoke and official
export/scalar audit passed; both implementations passed 64-env/5-iteration smoke.
Compiled-model equality, exact reset state/RNG parity and first-reset actor
observation equality are verified. Repeated original runs also show contact-force
and trajectory nondeterminism; long-run behavior is still unverified. Walking
controls and the complete sensor-refresh comparison are in progress. See the
[baseline audit](reports/official-baseline-audit-2026-09-08.md) for evidence and limits.


Walking runtime controls now also match compiled arrays and initial/subset actor
and critic observations; all four original/owned representative PPO smoke runs
completed. A published-policy rehearsal exposed a Walking scoring false positive:
standing almost still met the global RMSE threshold. Scoring v2 now requires signed
motion response in every commanded segment; the preserved published-policy trace
is reclassified `behavior_failed`. This changes acceptance scoring, not training
semantics. Six focused protocol/reset tests pass. See the baseline audit for the
measured commands, responses and remaining interpretation limits.


Official-first resource priority is now active: two owned MuJoCo learners and one
isolated original StandUp control are training on GPU 7. Three live Newton
learners are suspended with SIGSTOP, retaining process/checkpoint state for later
SIGCONT; they are separate from the three earlier stopped SB3 attempts. The
original control passed 4096-env capacity/export and normalized numerical parity
before this status update and is running its 15,000-iteration budget. Baseline
iteration time improved by about 1.5× after suspending extensions. No long-run
behavior success is claimed. Evidence and safe resumption identity records are
linked in the [baseline audit](reports/official-baseline-audit-2026-09-08.md).


CPU/BAM behavioral acceptance requires revalidation: the pinned CPU controller
matched DOF-friction constraints by joint ids, unlike the training Warp path.
The owned CPU adapter now uses actual DOFs while retaining native motor/sag and
friction formulas. Three real-physics/reset tests pass, including independent
Jacobian-force projection and unchanged motor torque. Old traces remain preserved;
corrected policy replay has since completed without restoring the missing behavior (see current baseline audit). This is an evaluation/CPU-load correction,
not evidence that training converged. See the baseline audit.


## Current video review

`outputs/previews/review-0908-01/index.html` links six existing clips with explicit
labels: default-recipe owned Walking/StandUp, the same Walking policy's CPU
rehearsal, published StandUp in MuJoCo/Newton, and the separate angular-penalty
diagnostic. The gallery does not conflate published-policy replay with own
training success. All six local video targets were checked.
