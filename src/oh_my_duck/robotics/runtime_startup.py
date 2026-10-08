from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from oh_my_duck.core.paths import project_root


class RuntimeStartup:
    def __init__(self, scene_id: str, robot_model: str):
        self.started = time.monotonic()
        self.path = project_root() / "outputs/runtime-startup" / str(uuid4()) / "startup.json"
        self.path.parent.mkdir(parents=True, exist_ok=False)
        self.record = {"schema_version": 1, "scene_id": scene_id, "robot_model": robot_model,
                       "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "started_at": datetime.now(timezone.utc).isoformat(), "ready": False, "stages": []}
        self.mark("imports")

    def mark(self, stage: str, *, ready: bool = False):
        elapsed = time.monotonic() - self.started
        stages = self.record["stages"]
        if stages:
            stages[-1]["duration_s"] = elapsed - stages[-1]["elapsed_s"]
        stages.append({"stage": stage, "elapsed_s": elapsed,
                       "at": datetime.now(timezone.utc).isoformat()})
        self.record.update(ready=ready, elapsed_s=elapsed)
        self.path.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"runtime_startup": stage, "elapsed_s": elapsed, "record": str(self.path)}),
              file=sys.stderr, flush=True)

    def reference(self):
        if not self.record["ready"]:
            raise RuntimeError("Runtime initialization has not completed")
        return {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "elapsed_s": self.record["elapsed_s"]}
