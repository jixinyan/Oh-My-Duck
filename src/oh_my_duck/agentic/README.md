# Application services and robot tools

| Responsibility | Source |
| --- | --- |
| Explicit application dependencies | [application.py](application.py) |
| Native task service interface | [harness/base.py](harness/base.py) |
| Native HTTP implementation | [../integrations/native_client.py](../integrations/native_client.py) |
| Tool definitions and argument validation | [tools/catalog.py](tools/catalog.py) |
| Simulation tool registration | [tools/simulation.py](tools/simulation.py) |
| Skill interface and lifecycle | [skills/base.py](skills/base.py), [skills/simulation.py](skills/simulation.py) |

`ApplicationServices.harness` accepts the `HarnessBridge` interface implemented
by `NativeTaskClient`. Its operations are `open`, `submit`, `status`, `wait`,
`stop` and `close`. The native server owns sessions, planning, tool registration,
execution and experience records. Run records preserve the server's identity,
execution domain and formal task state. `stop` waits for terminal execution and
requires device confirmation.

`submit(instruction, context_run_ids=...)` 接受同一打开会话中最多四项已结束
任务的不同编号。原生服务核查归属并生成后续任务的历史摘要；提交等待
任务收尾与 session 的 `ready` 状态，关闭要求确认资源释放。
Planner 通过原生 `skills.search` 和 `skills.load` 读取持久经验，
已关闭会话保持只读。

Voice uses this same implementation in `omd voice-task` and `omd voice-session`.
The [deployment guide](../../../docs/harness-native-integration.md) describes
the pinned native server and simulator environment.

Tool schemas use JSON Schema Draft 2020-12 through `jsonschema`, provided by
the `validation` extra and locked execution environments. Schemas are checked
at registration; arguments are checked before invoking a handler. Inputs must
contain finite JSON numbers. References, conditional requirements and array
constraints use the standard JSON Schema implementation. Simulation definitions
declare supported parameters, required values and additional-property rules.
The skill runner checks whole control ticks, task ownership and physical stop
confirmation.

Importing application interfaces loads no simulator or model. Actual service
clients and execution adapters import their dependencies when selected.
