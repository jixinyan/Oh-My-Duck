# 原生 context 设置与独立安装

固定实现源码：`bece543ff1cd4ed65ff3321ca6bdd54382cf2820`。
原生 SDK：`8a5e685b22d032207f53db20454f0992a4ad60fd`。

`integrations/edh/model-context.mjs` 的 `nativeContextManagement` 定义部署的
compaction 与图像历史设置，`server.mjs` 直接使用该设置。
原生 SDK 管理 headroom、摘要生成、历史保留和 agent loop；
部署使用 `thresholdRatio=0.7`、`retainRatio=0.15`、摘要输出上限
`8192` tokens 和最多十二张图像。headroom 使用原生 SDK 的默认
`4096` tokens，模型输出预算由公开参数独立设置。

部署的模型元数据声明上下文容量为 `32768` tokens。
Mac 与 Linux 均直接调用固定 SDK 的 `contextManagementOptions`、
`validateCompactionRoute`、`resolveConfig`、`resolveTargetPolicy` 和
`resolveCompactSpec`，验证实际部署设置。

| 模型输出预算 | Compaction 触发阈值 | 保留预算 |
| --- | --- | --- |
| 256 | 22,937 | 4,876 |
| 4,096 | 22,937 | 4,300 |
| 8,192 | 20,480 | 3,686 |

两个平台的六项检查一致，部署与 SDK 的四项输入均通过固定 Git 文件 SHA256
复核。摘要请求身份、原始任务权限、模型 reasoning 设置、任务目标、控制预算、
物理停止边界和正式 Verifier 继续使用对应原生实现。
这六项检查验证配置与原生预算计算；完整模型任务使用独立执行记录。

Mac 与 Linux 从干净固定源码构建并安装 wheel 和 source distribution。
每台主机均通过 444 个源码及资源文件、三份许可证和项目目录之外的
28 项实际 CLI 检查；112 项命令开始与退出记录完整，全部退出状态为零。
两个平台的构建文件分别通过 SHA256 复核。

证据：

- `outputs/acceptance/native-context-policy-mac-20261011-01/result.json`
- `outputs/acceptance/native-context-policy-linux-20261011-01/result.json`
- `outputs/acceptance/native-context-mac-20261011-01-install/`
- `outputs/acceptance/native-context-linux-package-20261011-01/`
- `outputs/acceptance/native-context-preparation-handoff-20261011-01.json`

完整 CPU 模型导航正在按原始 seed、办公室目标、出生位置、五个保持样本和
4,000-control/1,800-second 预算验收。正式任务结论、原始事件、模型响应、
实际动作、原始图像、停止确认和资源关闭使用该任务的独立记录。
当前准备交付记录保留 `model_navigation_acceptance_performed=false`。
当前 Newton 运行与 Microduck 设备具有各自验收要求；GPU 验收与 RL 保持停止。
