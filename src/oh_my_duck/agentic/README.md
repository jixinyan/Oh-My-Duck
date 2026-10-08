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
