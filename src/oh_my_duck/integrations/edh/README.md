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
environment reset. Verified policy inference and physical stepping remain on
the native device owner thread. Every admitted action retains its observation,
execution, generation and physical step identity.

`robotics/policies/catalogue.py` admits optional SHA256-verified training packages
alongside the pinned official policies. `joint_onnx.py` owns the common API-1 graph
and metadata checks. Scene configuration passes `policy_registry` to either
backend. A configured locomotion alias uses the same distance/angle controller;
registered policy selection preserves native boundary, action-scale and model
checks. See [package registration](../../../../docs/policy-registry.md).
