# 执行环境与资源管理

| 需要定位的功能 | 入口文件 |
| --- | --- |
| 官方源码、模型及隔离环境准备 | `bootstrap.py`、`bootstrap_isaac.py` |
| 安装的 OpenUSD 来源及文件检查 | `usd_runtime.py` |
| GPU 设备、显存、compute PID 与独占检查 | `gpu_inventory.py` |
| 本项目启动进程的取消与退出记录 | `owned_process.py` |
| 远程验收进程及控制通道 | `acceptance_supervisor.py` |
| 后端与原生训练入口选择 | `run.py` |
| EGL 与 headless 环境设置 | `headless.py` |
| W&B 配置及训练记录 | `tracking.py` |
| 源码、依赖与文件 SHA256 | `provenance.py` |
| 不可变任务源码副本 | `snapshot.py` |
| scheduler 任务提交 | `submit.py` |

`gpu_inventory.py` 使用 Python `csv` 读取原生 `nvidia-smi` 输出，提供不可变的
`GpuDevice` 和 `ComputeProcess`。设备记录检查完整字段、唯一身份、有限数值及显存范围。
进程记录保留原始行和重复 PID；独占检查拒绝请求设备上的任何 compute PID。

`validation/release/campaign.py` 负责连续状态采样、十秒零 utilization、16 GiB
可用显存、设备 2–4 的入口限制与验收阶段管理。`rl/experiments/scaling.py`
负责训练测量期间的显存和其他会话进程记录；`adopt_preparation.py` 使用同一模块
读取设备身份和忙碌进程。状态读取不会启动 CUDA 或分配 GPU 计算资源。

资源检查的实际记录及验证范围见[GPU 状态读取检查](../../../../docs/reports/gpu-inventory-2026-10-09.md)。
