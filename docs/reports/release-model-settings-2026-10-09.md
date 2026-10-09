# 发布与原生模型设置

固定源码：`fcef953523703afe22382ee31ec07deddb271aa9`。

`integrations/model_settings.py` 为原生 Harness、发布 controller 与远程发布前检查
提供同一份参数验证、API 名称处理、有效设置和启动参数。
模型标识拒绝空白和控制字符，API 与 reasoning effort 使用声明值，
输出预算必须为 256–8192 的整数。发布流程使用 8192 token 和 `high`，
支持明确选择模型及 API；原生 Harness 保留自己的默认预算。

发布前检查保存 `model_settings`，controller 保存并使用同一份设置启动导航。
`ModelSettings` 的元数据只包含模型标识、API、reasoning effort 和输出预算。
原生 `--check-model` 结果包含这四项实际运行设置及返回内容。

## CPU 入口与计划验证

117 项检查通过，覆盖共享模型设置、原生任务参数、完整发布计划和实际 GPU 记录。
八项实际发布 controller/worker CLI 调用验证预算拒绝，原生 Harness 的预算检查继续通过。
无效参数在读取 SDK、模型配置及创建输出目录前终止。
证据：`outputs/acceptance/model-settings-tests-20261009-02.xml`。

## 原生 GPU 查询期限

GPU 状态检查的三十秒期限覆盖设备查询、进程查询、采样间隔和最终接纳。
SSH 查询使用期限内剩余时间；期限耗尽后传播错误并保存中止记录。

实际 `jd_B300` 采样用时 24.39 秒，保存三次 GPU 2 设备与进程记录，
持续空闲与独占检查通过；GPU 4 上的两个现有 PID 被拒绝。
证据：`outputs/acceptance/gpu-inventory-live-20261009-02/result.json`。

实际 SSH 只读查询使用 0.01 秒测试期限，返回 `TimeoutExpired`，
中止记录保存为 `aborted`，没有 GPU 计算任务。
证据：`outputs/acceptance/gpu-query-timeout-20261009-01/result.json`。

## 完整发布前检查

相同固定版本的干净本地与远程源码完成六个阶段、四个场景、十个官方 policy、
29 个固定 Harness 文件、机器人资产、场景来源、公开地图与依赖版本检查。
controller 与远程结果中的有效设置一致：`gpt-6-luna`、`responses`、`high`、8192。
记录为 `preflight_passed`，没有分配设备或启动验收 worker。

证据：`outputs/acceptance/release-model-settings-preflight-20261009-01/`
中的 `campaign.json`、`preflight/result.json` 与 `independent-settings.json`。

## 独立安装

Mac 与 Ubuntu 22.04 独立构建安装通过 440 个 source/resource 文件与三份许可证。
Mac 完成 29 项 CLI 检查，Linux 完成 27 项检查。
证据分别位于 `outputs/acceptance/release-model-settings-20261009-01-install/result.json`
与 `outputs/acceptance/release-model-settings-linux-20261009-01/package-audit.json`。

## 实际原生模型调用

独立安装的 `omd harness --check-model` 使用用户指定私有配置中的 OpenAI 凭据，
通过固定原生 Harness SDK 调用 `gpt-6-luna`。
实际输出记录 `responses`、`high`、8192 token 与 `finish.kind=stop`，
传输记录包含 HTTP 200 和 `response.completed`。进程正常退出，退出状态为零。

原生 SDK 来源为 `8a5e685b22d032207f53db20454f0992a4ad60fd`，
从保存的 Git bundle 建立完整源码并按 `pnpm-lock.yaml` 安装依赖，跟踪源码保持干净。
凭据只通过进程环境传递。此调用检查模型接入与参数传递，不创建机器人会话。

证据：`outputs/acceptance/release-model-settings-native-20261009-03/`
中的 `result.json`、`process.json`、`stdout.json` 和 `data/model-transport.jsonl`。

独立交付检查重新核查测试 XML、两个安装报告、完整发布前记录、实际模型响应、
进程结束、所有原始采样 SHA256、五个改动模块的 Git/安装文件、固定 SDK 来源、
实际 `omd status` 输出和保留的资产依赖文件，结果见
`outputs/acceptance/release-model-settings-handoff-20261009-01.json`。

本次工作验证 CPU 代码、实际设备状态查询和发布执行准备。
GPU 运行、RL 学习行为与 Microduck 真机保留对应的独立验收要求；GPU 验收与 RL 保持停止。
