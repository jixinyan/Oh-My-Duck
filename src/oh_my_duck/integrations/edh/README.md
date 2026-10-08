# Native EDH worker source map

| Responsibility | Implementation |
| --- | --- |
| Simulation initialization, observations, action stepping and native checks | [environment.py](environment.py) |
| Owner-thread action execution and stop acknowledgement | [device.py](device.py) |
| Native session lifecycle, control tools, metric motion and ActionGate boundaries | [session.py](session.py) |
| JSONL process transport, request dispatch and disconnect cleanup | [worker.py](worker.py) |
| Stable executable module and class imports | [../edh_native.py](../edh_native.py) |

The server launches `python -m oh_my_duck.integrations.edh_native` locally or
through SSH. CPU MuJoCo/BAM and Isaac Newton/BAM use the same native session,
tool operations and physical evidence. Importing these modules requires the
pinned `physical_harness` package; selecting Isaac initializes its runtime during
environment reset. Official policy inference and physical stepping remain on
the native device owner thread. Every admitted action retains its observation,
execution, generation and physical step identity.
