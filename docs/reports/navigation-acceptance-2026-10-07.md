# 原生导航验收 — 2026-10-07

## Hospital 轮滑路线

真实 `gpt-6-astra` Responses/high 通过固定原生 Harness、ActionGate、官方 `roller`、Newton SolverMuJoCo/MuJoCo-Warp、BAM 与独立 Verifier 完成 Hospital 路线。机器人从 `(7.96,7.15)` 出发，按顺序进入东向 checkpoint `(9.7,7.2)`、北向 checkpoint `(9.7,9.4)`，到达西侧目标 `(8.1,9.4)`。任务保持同一个物理 episode，模型根据当前 odometry、head RGB、ToF 和明确授权的 simulator ground truth 选择工具参数。

| 测量 | 结果 |
|---|---|
| 原生任务 | `cf059bda-3f7e-4294-93e7-8b5fe9d70df1`，`succeeded` |
| 独立 Verifier | `b2343534-6b50-4105-b98b-3982b7fef7ad`，`passed` |
| 行走段端点位移累计 | 5.438808 米 |
| 工具调用 | 3 次 `walk`、4 次 `rotate`、8 次当前图像的 GT 感知 |
| checkpoint 首次访问 | 控制步 309、1025 |
| 最终位置 | `(8.072023,9.308546)` 米 |
| 最终目标误差 | 0.095637 米，要求 ≤0.3 米 |
| 最终保持 | 75 个零命令控制步，80 个连续停止样本 |
| 执行计数 | 1903 个 policy call / 控制步，7612 个物理子步 |
| 外部障碍接触累计 | 零 |
| 原始记录 | 3130 个事件、393 张图片，其中 380 张 observer 图片 |
| 会话 | `c231eb6e-cbab-4e65-8b51-22cca522a668`，`closed` / `released` |

| 原始请求 | 实测值 | 目标误差 | 动作结果 |
|---|---|---|---|
| `walk(1.70)` 米 | 1.675291 米沿初始朝向位移 | 0.056756 米 | `failed` |
| `rotate(85.5)` 度 | 78.245093 度 | 7.254907 度 | `failed` |
| `rotate(10)` 度 | 13.931938 度 | 3.931938 度 | `complete` |
| `rotate(-10)` 度 | −11.119303 度 | 1.119303 度 | `complete` |
| `walk(2.10)` 米 | 2.071786 米沿初始朝向位移 | 0.040866 米 | `complete` |
| `rotate(91.8)` 度 | 95.351106 度 | 3.551106 度 | `complete` |
| `walk(1.65)` 米 | 1.690737 米沿初始朝向位移 | 0.040907 米 | `complete` |

动作验收保持 0.05 米 / 5° 和五个实际停止样本。导航目标验收独立检查路线和最终姿态。模型读取失败动作的最终位置与角度，继续完成后续路线。旋转伴随实际平移，模型在 westward turn 的 5.6 厘米平移之后重新计算行走距离。ToF 的 invalid 返回和空目标查询保留原始结果。

## 来源与复核

| 来源或产物 | 身份 |
|---|---|
| 不可变 worker 源码 | `4feebc37dadd45cb487305117c8a6a2f421951b2` |
| Newton backend SHA256 | `5449afbebf98da3a655f53c665b28a476e12f04a8b8b1b82a1b26d665f9fd883` |
| metric controller SHA256 | `8eb6153fcd15d9287fd1d6f35d4cc96341d5f9e1518188a1e8a74f0503cc37db` |
| 原生 Harness | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy catalog | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| `roller.onnx` SHA256 | `cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd` |
| 主机 / physical GPU | `jd_B300` / 4 |
| seed | `20261007` |
| 求解器 | 10 iterations、20 line-search iterations、CUDA graph |
| 原始导出目录 | `outputs/demos/hospital-navigation-20261007-02` |
| 独立复核文件 | `outputs/acceptance/hospital-navigation-20261007-02.json` |
| 独立复核 SHA256 | `10d12034502537dca06cd1ce54b34bcec7760ff114ae1a684fcd8eb7c28e61e0` |
| manifest SHA256 | `e753a809ccb2338c2bdf3071836272f573d7522d36bed4af8bb7823d066e4451` |
| run JSON SHA256 | `2807dcb7efb5b6587a5506d3f3cf0ed233d184ed9daa7934c8bd6680d7f2b10a` |
| events JSON SHA256 | `67af90ff99aa864f166ea5b3a6a66cdebaa6699adeaca8787fd8dfceb634d637` |
| frames JSON SHA256 | `d09acd87b09b93fedd0c8b9bc9aa654d5e6e5a663ba1754913c800467ded004c` |

`accept_navigation_replay.py` 根据原始工具参数、起始朝向和物理端点重新计算行走误差；旋转检查记录的累计 yaw。全部动作具有终态测量，已完成动作满足原有精度和停止要求。检查程序验证 checkpoint 顺序、最终姿态、native `policy_stop`、Verifier 的物理证据、控制计数、原始 PNG 字节与尺寸、observer 物理时间、十四个 servo 的实际传感器测量，以及累计零外部接触。

原始图像和事件在会话关闭后通过原生持久记录导出。当前自动入口 `run_navigation_acceptance.py` 创建任务、保存状态、等待终态、关闭会话、确认资源释放、导出记录并执行独立复核；成功文件只在全部要求通过之后生成。操作见[导航验收说明](../navigation-acceptance.md)。

## Agentic MP4

`outputs/demos/oh-my-duck-hospital-navigation-20261007.mp4` 为 170 秒、1920 × 1080、10 fps、1700 帧的视频，包含实际 Newton/Warp 的 observer 场景相机、head RGB、公开 Planner 文字、计划、工具参数、运动误差与独立 Verifier 的结果。模型等待时间按 16× 压缩，全部 380 张 observer 图片保留相邻帧的实际物理时间间隔。图片来源、视频时间、每帧文字范围和完整 FFmpeg 解码通过检查；公开 trace 没有 private reasoning。

MP4 SHA256：`29977ffbeca444e30047b1dd205867f2e2c8dbd0fde029ba390189e0708267a5`。同名 JSON 保存来源 SHA256、相机与事件数量、播放时间安排和编码参数。

验收范围为指定 seed、出生位置和授权 GT 信息下的路线。行走段端点位移累计与连续路径长度分别定义。长距离动作的全部请求精度、图像输入独立 VLN、识别准确率、多场景泛化、物体携带、自训练 policy 行为和真机仍需独立验收。RL 保持停止。
