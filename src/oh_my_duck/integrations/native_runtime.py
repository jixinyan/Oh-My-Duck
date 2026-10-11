from pathlib import Path
import subprocess


EDH_REVISION = "8a5e685b22d032207f53db20454f0992a4ad60fd"


def verify_edh_source(directory: Path) -> Path:
    source = directory.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != EDH_REVISION:
        raise ValueError("EDH source revision differs from the pinned native runtime")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed:
        raise RuntimeError("Pinned EDH source contains tracked changes")
    for name in ("package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.runtime.json",
                 "harness/contracts/schema/physical.schema.json", "apps/server/src/index.ts",
                 "harness/contracts/src/index.ts", "harness/agent-runtime/models/src/index.ts"):
        (source / name).resolve(strict=True)
    return source
