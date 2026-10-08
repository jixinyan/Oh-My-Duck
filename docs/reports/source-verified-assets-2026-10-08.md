# Source-verified converted assets and release preparation

`omd assets --backend isaac-newton -- --reuse-from-source CHECKOUT` explicitly
reuses generated Microduck assets after verifying their complete conversion
inputs and provenance. The implementation is
`src/oh_my_duck/rl/backends/isaac_newton/asset_reuse.py`; its responsibilities and
usage are in the [backend source map](../../src/oh_my_duck/rl/backends/isaac_newton/README.md).

## Verification requirements

The current and baseline sources must be clean complete Git checkouts. The
baseline revision must be an ancestor of the current revision. Verification
covers robot sources, conversion tools, Python bootstrap modules, upstream pins,
conversion environment configuration and the complete resolved dependency lock.
Project optional-dependency declaration metadata is excluded only when the
project has no runtime dependencies and the conversion environment activates no
project extra. Every other lock field participates in the comparison.

The baseline asset passes the baseline checkout's standard `require_asset`
verification. The complete generated file inventory must match its manifest,
all SHA256 values must match, and symbolic links and paths outside that directory
are rejected. Files are copied into a temporary directory under the generated
asset parent and verified before the new destination is published. Source
revisions and fingerprints are checked again before publication. Existing
destinations are rejected.

`source_reuse` records both Git revisions, both fingerprints, the baseline build
SHA256, every conversion-input SHA256, the resolved dependency hash and verified
file count. The original conversion and physical-validation fields are preserved.
The current checkout's unchanged `require_asset` validates the completed result.

## Actual remote execution

Both public CLI calls ran on `jd_B300` from immutable source
`bf8abf881f57880a352fcb765f1182f80b710bb7`, using baseline checkout
`a3cbcc160ca9bb25a12e350d0125e877e53b0934`. Both checkouts retained clean tracked
source. CUDA visibility was empty, and Kit and simulation workers were not started.

| Model | Conversion inputs checked | Generated files checked | Preserved status |
| --- | --- | --- | --- |
| `allcollisions` | 129 | 11 | `converted_not_physics_validated` |
| `groundcontact_rollers` | 129 | 11 | `converted_not_physics_validated` |

An independent audit compared every original/copied generated file, every input
hash, dependency documents and the complete build manifests. Standard current
and baseline asset lookups passed. Actual repeated public CLI calls rejected
existing destinations, and original build files remained unchanged.

Evidence is under `outputs/acceptance/asset-reuse-20261008-01/`:
`allcollisions.log`, `groundcontact_rollers.log`, and `independent-audit.json`.
The resolved dependency document SHA256 is
`7fed5c57709c5d85cefb141ed792e87578ce5eb39c7ae5d13de1444023d8da8b`.
Current source fingerprints are
`1c79ad78b397f746b9e64da49ca9c5aa41c05ef3f3a048d557cd7aa479fe750b`
for `allcollisions` and
`e2f84f631c96beb71f6852c0b69dd29d4adf0c70bd3a67fe3ee39487e04bf31e`
for `groundcontact_rollers`.

## Complete metadata and installed-package checks

The matching local/remote immutable source passed the complete release metadata
preflight: six stages, four scene configurations, nine declared input hashes,
29 pinned native Harness source files and all ten official policies. Each Office
configuration verified 2291 resource files; each Hospital configuration verified
1639 files. Robot asset fingerprints, generated-file hashes, public map identity,
environment versions and provider metadata passed. The result records
`cuda_runtime_checked=false`. Evidence is
`outputs/acceptance/asset-reuse-preflight-20261008-01/preflight/result.json`.

A freshly built wheel and source distribution were installed into an independent
environment. Actual package checks verified 418 tracked Python/resource files,
three licenses and 19 installed CLI calls outside the checkout. This includes
independently recomputed 1208 recorded ONNX actions with zero maximum error,
1220 serialized native updates and the complete saved five-motion physical
audit. Installed asset-module import and `--reuse-from-source` help also passed.
Evidence is `outputs/acceptance/asset-reuse-installed-package-20261008-02.json`
with its companion policy and physical audit reports and execution log.

These checks cover artifact preparation, real saved evidence and installed
execution with CUDA disabled. Fresh Newton physics, kernel compilation, model
behavior across scenes and hardware require their own actual acceptance.
GPU acceptance and RL remain stopped.

| Artifact | SHA256 |
| --- | --- |
| Independent asset audit | `396deaa4b29685d223e809497fd5d91064c869b9b2d839fd55fc069e707d0e00` |
| Complete metadata preflight | `ff4268fe488939f1e28f0d19164efbb8b3ed23693fe0e6e625c05478ae4d386c` |
| Independent installed package audit | `812199297b0a04c7a22af919bda2f22afa54f8ff7bcaa002bfd45367f71bbdac` |
| Wheel | `b81415b4e2f9fb441463e4bc3c55079ee652d06e73d1627460ab5c86e162d946` |
| Source distribution | `39f40c96d1110925c58071e24c6ddd8c7b379109f2ad1642e30e14a963bcf581` |
