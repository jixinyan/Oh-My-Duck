# 原生 Responses 工具参数与模型调用

固定实现源码：`246f83f1deb6c68a4f3421d529e6283aa8094302`。
原生 SDK：`8a5e685b22d032207f53db20454f0992a4ad60fd`。

`integrations/edh/model-transport.mjs` 为原生 Responses function tools 明确设置
`strict: false`，保留 SDK 定义的 properties、required 和 additionalProperties。
工具已经声明的 strict 设置保持原有值，原生 SDK 继续验证调用参数和执行权限。
其他请求通过标准 `Request` 接口传递正文。关闭模型连接时恢复 fetch 和
Undici dispatcher，并等待当前 dispatcher 关闭。
Responses 的参数规则见[官方 function calling 文档](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)。

实际模型为 OpenAI `gpt-6-luna`，使用 Responses、`high` reasoning 和
`8192` 输出预算。两个实际模型响应均完成：模型调用原生 `team.query` 读取
已保存的正式 Verifier 报告，随后返回报告中的 status 和 verdict ID。

| 检查 | 实际结果 |
| --- | --- |
| API 返回的工具定义 | `strict=false`，required 仅包含 `assignmentId`；完整参数 Schema 与 SDK 一致 |
| 模型生成的参数 | 仅包含 `assignmentId`，通过原生参数验证 |
| 正式报告 | `0fc6b032-5baf-4f54-a13e-3f775fb956a1` |
| 正式 verdict | `282e6bd9-c406-4d1b-af59-06bc1c47c876`，status 为 `passed` |
| 原始记录 | SHA256 保持 `434a62790b04d0b6000d0737150d1a2d054903f29381c48fddd9b8fb47b1e1ea` |
| 传输关闭 | fetch 与 dispatcher 恢复，连接关闭检查通过 |

Mac 与 Linux 从该固定源码构建并安装 wheel 和 source distribution。
每台主机通过 444 个源码及资源文件、三份许可证和项目目录之外的
28 项实际 CLI 检查。独立复核确认 112 项命令开始与退出记录完整，
所有退出状态为零；用户的 Isaac asset lock 文件 SHA256 保持原有值。

证据：

- `outputs/acceptance/native-tool-parameters-linux-20261011-03/`
- `outputs/acceptance/native-tools-mac-package-20261011-02/`
- `outputs/acceptance/native-tools-linux-package-20261011-02/`
- `outputs/acceptance/native-tools-handoff-20261011-01.json`

该检查使用真实模型和已保存的正式报告，记录保留
`model_navigation_acceptance_performed=false`。
完整 CPU 导航使用两个既定 seed、办公室目标、出生位置、五个保持样本和
4,000-control/1,800-second 预算，依据独立任务结论、物理记录和资源关闭验收。
Newton 与 Microduck 设备具有各自的验收要求；GPU 验收与 RL 保持停止。
