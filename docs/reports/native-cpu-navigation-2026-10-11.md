# Luna 原生 CPU 重复导航验收

固定实现源码：`7b94665bb116aeb411597f67c459b4f19647e901`。
原生 SDK：`8a5e685b22d032207f53db20454f0992a4ad60fd`。
官方 policy：`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。

Mac 上的原生服务使用 OpenAI `gpt-6-luna`、Responses、`high` reasoning
和 `8192` 输出预算。`jd_B300` 的 SSH worker 使用 CPU MuJoCo/BAM、
OSMesa 和官方 Alpha Walking policy，实际 CUDA visibility 为空。

两项任务使用相同指令：“请使用距离和角度工具前往办公室。读取相机和距离传感器，到达后停止。”
出生位置为 `(0, 0, 0)`，原生目标为办公室区域、
upright 条件及五个保持样本；每项任务保留 4,000-control/1,800-second 预算。
原生 Harness 负责规划、工具权限、ActionGate 和独立 Verifier。

| seed | control steps | physics steps | events | observer frames | stop-progress records | model responses |
| --- | --- | --- | --- | --- | --- | --- |
| 20260929 | 583 | 2,332 | 1,110 | 116 | 17 | 59 |
| 20260930 | 828 | 3,312 | 1,551 | 165 | 20 | 78 |

两项任务的状态均为 `succeeded`，正式 Verifier 均为 `passed`，执行以
`policy_stop` 结束。独立安装的 `omd validate replay` 核验米制运动目标、
实际位置变化、五个停止样本、连续事件、PNG 来源、物理计数和正式任务结果。
全部 137 个模型响应完成，HTTP 状态均为 200。

| seed | run ID | verdict ID |
| --- | --- | --- |
| 20260929 | `dacfcdb4-c7aa-471d-b89d-8f3fee5f23b4` | `37543995-7bea-4fea-b83a-80553c5df3e2` |
| 20260930 | `a9073ac6-7fed-4928-a0ab-1a97bd8c858d` | `75545b22-a69d-414e-8a33-8171b123f8c2` |

两次 session 关闭均确认 `state=closed` 和 `resources=released`。
所属服务及独立复核进程均正常退出；任务结束后的远程检查确认本次 worker
数量为零。用户的 Isaac asset lock 保持原有 SHA256。

证据目录：`outputs/acceptance/native-tools-navigation-mac-ssh-20261011-03/`。
独立结果：`outputs/acceptance/native-cpu-navigation-handoff-20261011-01.json`。
完整原始 JSONL、事件、相机图像、模型连接记录和进程资料均保留。

验收范围为该固定实现、官方办公室场景、Alpha Walking 和两个既定 seed。
多场景长导航、独立图像导航、Newton 运行与 Microduck 设备继续使用各自的
验收要求。GPU 验收与 RL 保持停止。
