# Harness integration review · 2026-09-30

The Oh My Duck checkout is aligned with `origin/main` at `ac6bf86`. The local
Embodied-DeepSeek-Harness checkout is on `96d0a17`, its 0.2.0 rc.2 review
checkpoint. Its current integration boundary is clear even though the user
facing wire transport is not ready: DSH owns sessions, model turns, native tool
registration and cancellation; EDH adds physical contracts, execution status,
evidence references and post-execution verification. `AbortSignal` is the
cooperative cancellation boundary for future physical providers.

Oh My Duck now has a deterministic protocol peer in
`oh_my_duck.agentic.harness.mock`. A caller supplies exact text routes to
registered local tools. The mock checks request identity and idempotency,
validates tool arguments against the catalog's supported JSON Schema subset,
deduplicates event and experience publication, and reports unsupported routes
without claiming success. It does not create a model session, planner,
scheduler, memory store or autonomous loop.

The bridge contract carries the stable semantic fields already required by the
project design: session/request identity, optional recorded-audio reference,
tool definitions, episode events, evidence references and cancellation. The
`HarnessEndpoint` metadata records an eventual deployment endpoint and source
revision without hard-coding an EDH wire format. A real adapter remains
explicitly unavailable until EDH publishes a stable transport binding.

Validation on 2026-09-30:

- `PYTHONPATH=src python -m unittest tests.test_harness tests.test_framework -v`: 9 passed.
- `python -m compileall -q src/oh_my_duck/agentic tests/test_harness.py`: passed.
- `git diff --check`: passed.

The local system still has no concrete RobotBackend, SkillRunner, microphone or
playback adapter. The next execution milestone is policy validity: Walking and
StandUp need reproducible cross-backend execution evidence from the declared RL
acceptance workflow. Only verified policies should then be exposed through a
simulation-side or deployment tool adapter, with the same cancellation and
evidence semantics, followed by a transport adapter once EDH's API is ready.
The mock is protocol evidence only; it is not an autonomous Harness acceptance
result.
