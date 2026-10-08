# Native CPU pose verification

The public `omd validate pose` campaign passed at immutable source
`9b9c22e8cf82ebdab6a71ceb14745a0926f2e904`. It executed official `velstand` and
`alpha_stand` through the pinned native Harness, control tools and ActionGate in
the original furnished CPU MuJoCo/BAM apartment. Reproduction, requirements and
declared criteria are in [native pose validation](../native-pose-validation.md).
Implementation is in `validation/harness/pose.py` and `pose_records.py`.

## Physical measurements

The campaign completed 575 controls and 2300 physics substeps. Five pose phases
each executed 100 unique controls after the 75-control warmup. All command
slots and their native admission identities matched the recorded official ONNX
inputs. The audit checked 587 control publications, including 12 matching
publications at existing physical boundaries.

| Phase | Head pitch command, rad | Body height command, m | Mean measured head pitch, rad | Mean measured body height, m |
| --- | --- | --- | --- | --- |
| Neutral | 0 | 0 | 0.420504 | 0.116411 |
| Positive | +0.2 | +0.01 | 0.568634 | 0.118725 |
| Return positive | 0 | 0 | 0.465617 | 0.116642 |
| Negative | −0.2 | −0.01 | 0.280911 | 0.114272 |
| Return negative | 0 | 0 | 0.413506 | 0.115511 |

Means use the final 25 unique controls of each phase. Positive changes were
0.148130 rad and 0.002315 m; negative changes were −0.184706 rad and −0.002370 m.
Both directions passed the declared 0.03 rad / 0.001 m minimum response. Each
zero-command return reduced error relative to the preceding neutral pose.
Endpoints remained upright, external obstacle contacts remained zero, and
native termination confirmed 95 consecutive stopped samples.

## Records, installation and resource release

The software renderer was `llvmpipe (LLVM 15.0.7, 256 bits)`. The audit decoded
all 115 observer PNG frames, compared their original native event bytes and
timestamps, and checked head RGB and named servo readings at every pose boundary.
CUDA remained uninitialized. The session released its policy server, control
server and action device; the campaign exited with status zero. The final
process audit found no remaining pose campaign process.

Linux recomputation of all 575 official ONNX actions had maximum error zero.
The fresh installed command also passed outside the checkout on macOS; its
maximum action error was `5.364418029785156e-7`, within the unchanged `1e-6`
policy parity tolerance. Source/configuration/import checks passed 30 tests.
A fresh wheel and source distribution verified 425 source/resource files and
22 CLI calls outside the checkout. Current-source remote metadata verification
passed all six release stages, four scenes, ten policies and 29 pinned Harness
files without CUDA execution.

| Evidence | Path |
| --- | --- |
| Complete native campaign and measurements | `outputs/acceptance/native-pose-20261008-06/result.json` |
| Original actions, publications and camera frames | Same directory: `samples.json`, `events.json`, `tools.json`, `frames.json`, `frames/` |
| Installed independent physical audit | `outputs/acceptance/native-pose-installed-audit-20261008-07.json` |
| Fresh package verification | `outputs/acceptance/native-pose-package-20261008-06/result.json` |
| Current-source release preparation | `outputs/acceptance/native-pose-release-preflight-20261008-06/preflight/result.json` |
| Configuration and dependency isolation | `outputs/acceptance/native-pose-imports-20261008-06.log` |
| Final process audit | `outputs/acceptance/native-pose-process-audit-20261008-06.log` |

| Artifact | SHA256 |
| --- | --- |
| Complete native campaign | `b2b847995bdf65ce172845a8ba89adcf1f8b6da04590e2119e4620fc3b762be1` |
| Installed pose audit | `dba4c6459962ccc7256b0284dc167f5780e7563bcd4f45532fb1dacef36628f3` |
| Package verification | `c7d8e04185fa136ec11fc7d88472321ad948d1682c9fb87e4a43c9d12def45f0` |
| Remote release preparation | `f77f748f178ee1f51b2e32da7b6b5725395f83d800fbba4f257c7a5830e40218` |
| Wheel | `3a83cf762af64982ddbb263abe254e44e4696fe4b63ef70c92b28380d78bac0e` |
| Source distribution | `b04a5465ef9eadf9d7a85c784e2f931dc13380c81e9c2057f6097b03aa8a5cea` |

The verified behavior scope is the coupled head pitch/body height schedule
with `alpha_stand` in the CPU apartment. Other pose axes, precise pose tracking,
current-source Newton execution, full model-driven tasks and hardware each have
their own acceptance scope. GPU acceptance and RL remain stopped.
