# Native Harness 接入

| 功能 | 入口 |
| --- | --- |
| 会话、任务提交、历史引用、等待与停止 | `native_client.py` |
| 共享模型标识、API、reasoning effort 与 token 预算 | `model_settings.py` |
| 物理环境、设备、工具、ActionGate 与 worker 通信 | [edh/README.md](edh/README.md) |
| 保留的原生 worker 模块入口 | `edh_native.py` |
| Node 服务、原生模型调用与传输记录 | 仓库根目录 [integrations/edh/server.mjs](../../../integrations/edh/server.mjs) |

`model_settings.py` 由 `cli/harness.py`、`validation/release/campaign.py`
和 `validation/release/worker.py` 共同使用。参数在访问模型配置或 SDK 前验证，
配置文件中的 `wire_api` 只能使用声明的值。发布前检查保存有效设置，
后续导航通过同一设置构建启动参数。模型元数据只包含四项运行设置。

原生 Harness 管理 agent loop、经验与正式任务验证。此目录提供 HTTP 接入、
Microduck 物理执行和模型设置；凭据由私有配置或环境变量提供。
使用方式与实际检查见[发布模型设置](../../../docs/reports/release-model-settings-2026-10-09.md)。
