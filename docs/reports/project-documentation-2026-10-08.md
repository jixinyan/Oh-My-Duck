# Project documentation — 2026-10-08

Verified baseline: `86e59271bad9606bb8820dd77360a8f7dd45897f`.
Production source, task configuration, models, policies and runtime behavior
remain unchanged in this documentation milestone.

## Maintained documents

Project Design version 0.3 records the complete product scope, 13 modules,
actual native tools, voice interfaces, sensor formats, policy extension,
official observation/action timing, training/export requirements and acceptance.
Execution Plan version 0.3 retains steps 00–16 with dependencies, work,
deliverables and passing conditions. Implementation status provides the CLI
maturity matrix and links each capability to its recorded execution scope.
Architecture, development and getting-started guides identify current source,
CPU entry points and explicit GPU allocation conditions.

GPU acceptance and RL remain stopped. Future authorized execution uses at most
one idle GPU 2–4, with zero compute PIDs and sustained zero utilization.
Hardware, learned behavior, recognition accuracy, voice quality and
current-source Newton execution retain their separate requirements.

## Actual checks

MarkdownIt and its front-matter plugin parsed all six documents; PyYAML parsed
both versioned metadata blocks. All 111 local file targets exist. The 13 documented
native tools match the Node declarations and Python worker operations. All eight
documented VoiceProfile fields match the actual dataclass.

The independently installed CLI executed `status`, `harness --help`,
`voice-task --help`, `voice-session --help` and `validate release-plan`
outside the checkout. All five exited successfully. The release-plan check
covered six stages and four scenes with CUDA disabled; its nine input hashes
are retained. The protected local lockfile change remained unchanged.

Retained evidence: `outputs/acceptance/project-documentation-20261008-01/result.json`.
SHA256: `fdbeb20cf1a9480bbe5e52ee95fe7720cd568520a209d17e5ad1d6780dd45d84`.
The same directory contains all five actual command outputs.
GPU acceptance was not performed.
