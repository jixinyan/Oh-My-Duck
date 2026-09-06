# Framework validation — 2026-09-06

Scope: whole-project boundaries and lightweight implementations. Hardware, policy quality, Isaac and simulator training are not covered by this report.

## Checks

```text
python -m unittest discover -s tests -v
Ran 8 tests — OK

python -m compileall -q src scripts training omd.py
exit 0
```

The tests cover:

- Importing application, contracts, backends, skills, tools, perception, policies, harness, voice, recording and training interfaces without importing MuJoCo, Torch, Warp or Isaac Lab.
- Rejecting an unimplemented Isaac/Newton backend without selecting another physics engine.
- Returning unsupported for an unregistered tool.
- Rejecting duplicate tool registration and a handler result associated with the wrong request.
- Retaining simulation/real identity, cancellation status and evidence references in JSONL events.
- Exact command-segment boundaries at 50 Hz; rejection of fractional control ticks, negative durations, NaN commands and incompatible policy frequency.

The simulation/real records in these tests are schema fixtures, not physical-robot executions.

## Manual CLI checks

- `python omd.py --help`: public commands and architecture entry point.
- `python omd.py status`: software component maturity, distinct from live robot capabilities.
- `python omd.py submit --name omd-dry-run --dry-run -- python omd.py probe --output outputs/probe`: explicit one-node/one-GPU command; no submission.
- `python omd.py train --backend isaac-newton`: expected exit 2 with an unavailable message.

Environment: server entry host, Python 3.13.13. No simulator import or GPU work is required by this validation.

## Remaining work

Official Python 3.12 dependency setup and worker checks; exhaustive source audit; concrete robot/voice/harness adapters; Newton migration and sim2sim evidence. Type protocols establish extension points but do not enforce every runtime/wire constraint; concrete adapters need their own validation and behavior tests.
