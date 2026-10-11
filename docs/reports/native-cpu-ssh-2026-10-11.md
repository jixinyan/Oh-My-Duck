# SSH CPU worker 与资源关闭

固定实现源码：`7b94665bb116aeb411597f67c459b4f19647e901`。
原生 SDK：`8a5e685b22d032207f53db20454f0992a4ad60fd`。

`integrations/edh/server.mjs` 为 SSH CPU worker 明确设置
`CUDA_VISIBLE_DEVICES=''`。实际仿真在 `jd_B300` 使用 CPU MuJoCo/BAM
和 OSMesa，模型连接可以由 Mac 上的原生服务承担。

| 检查 | 实际结果 |
| --- | --- |
| 启动前的本次 worker | 0 个 |
| session 打开后的本次 worker | 1 个，实际进程环境中的 CUDA visibility 为空 |
| session 关闭 | `state=closed`，`resources=released` |
| 关闭后的本次 worker | 0 个 |
| 所属服务 | 正常退出，exit code 为 0 |
| Mac 与 Linux 独立安装 | 每台主机 444 个文件、三份许可证和 28 项 CLI 调用通过 |
| 独立复核 | 112 项命令开始与退出记录完整，用户的 Isaac asset lock 保持原有值 |

证据保存在 `outputs/acceptance/native-cpu-ssh-lifecycle-20261011-01/`、
`outputs/acceptance/native-cpu-ssh-mac-package-20261011-01/` 和
`outputs/acceptance/native-cpu-ssh-linux-package-20261011-01/`。
独立结果为 `outputs/acceptance/native-cpu-ssh-handoff-20261011-01.json`。

该检查记录 `model_navigation_acceptance_performed=false` 和
`gpu_acceptance_performed=false`。完整导航使用原始目标、预算和独立 Verifier
验收；GPU 验收与 RL 保持停止。
