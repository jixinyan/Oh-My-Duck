"""Bind a packaged ONNX to the native checkpoint that actually produced it."""

import hashlib
import json
from pathlib import Path


def verify_export_source(onnx, checkpoint):
    onnx, checkpoint = Path(onnx).resolve(), Path(checkpoint).resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report_path = onnx.parent / "export.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("framework") == "sb3":
            if Path(report["source_run"]).resolve() != checkpoint.parent or checkpoint.name != "model.zip":
                raise ValueError("ONNX source run differs from packaged checkpoint")
            if report.get("policy_sha256") != sha(onnx):
                raise ValueError("Exported ONNX hash differs")
            run = json.loads((checkpoint.parent / "run.json").read_text())
            if run["files"]["model.zip"] != sha(checkpoint):
                raise ValueError("Native checkpoint hash differs")
            return report
    report_path = onnx.with_suffix(".provenance.json")
    if not report_path.exists():
        raise ValueError("Re-export through omd export to produce checkpoint provenance")
    report = json.loads(report_path.read_text())
    if Path(report["source_checkpoint"]).resolve() != checkpoint or report.get("checkpoint_sha256") != sha(
        checkpoint
    ):
        raise ValueError("ONNX source checkpoint differs")
    if report.get("policy_sha256") != sha(onnx):
        raise ValueError("Exported ONNX hash differs")
    return report
