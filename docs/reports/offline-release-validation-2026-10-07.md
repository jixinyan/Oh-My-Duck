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

发布参数检查共 16 项通过，使用正式六阶段计划和实际场景配置，覆盖重复阶段编号、非有限数值、来源路径、模型、设备、renderer、出生位置和预算类型。`doctor --metadata-only` 与 `run_release_campaign.py --preflight-only` 提供无需 CUDA 的远程检查入口，检查记录与行为验收分别保存。
