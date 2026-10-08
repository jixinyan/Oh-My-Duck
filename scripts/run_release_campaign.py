import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import re
import signal
import socket
import subprocess
import sys
import time

from jsonschema import Draft202012Validator

from accept_metric_campaign import PLAN_SCHEMA
from metric_plan import case_motions
from record_harness_demo import request, save_json


STAGE_SCHEMA = {
    "type": "object", "required": ["schema_version", "stages"], "additionalProperties": False,
    "properties": {"schema_version": {"const": 1}, "stages": {"type": "array", "minItems": 1,
        "items": {"oneOf": [
            {"type": "object", "required": ["id", "kind", "plan", "suite"], "additionalProperties": False,
             "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"},
                 "kind": {"const": "metric"}, "plan": {"type": "string"}, "suite": {"type": "string"}}},
            {"type": "object", "required": ["id", "kind", "scene", "instruction", "minimum_distance_m"],
             "additionalProperties": False, "properties": {
                 "id": {"type": "string", "pattern": "^[a-z][a-z0-9-]*$"}, "kind": {"const": "navigation"},
                 "scene": {"type": "string"}, "instruction": {"type": "string"},
                 "minimum_distance_m": {"type": "number", "exclusiveMinimum": 0}}},
        ]}}},
}


def ssh(args, command, **kwargs):
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                           args.worker_host, command], **kwargs)


def within(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Acceptance input must belong to the source checkout")
    return path


def gpu_snapshot(args, output, name):
    deadline = time.monotonic() + 30
    sample = 0
    idle_since = None
    while True:
        raw = ssh(args, "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
                  check=True, capture_output=True, text=True).stdout
        (output / f"{name}-{sample:02d}.csv").write_text(raw)
        rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
        selected = [row for row in rows if int(row[0]) == args.gpu]
        if len(selected) != 1 or float(selected[0][4]) - float(selected[0][3]) < 16384:
            raise RuntimeError("Allocated GPU requires at least 16 GiB available")
        processes = ssh(args, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader",
                        check=True, capture_output=True, text=True).stdout
        (output / f"{name}-{sample:02d}-processes.csv").write_text(processes)
        occupied = [row for row in csv.reader(processes.splitlines(), skipinitialspace=True)
                    if row[0] == selected[0][1]]
        if occupied:
            pids = sorted({int(row[1]) for row in occupied})
            raise RuntimeError(f"Allocated GPU already has compute processes {pids}; exclusive use is required")
        now = time.monotonic()
        if float(selected[0][2]) == 0:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= 10:
                return
        else:
            idle_since = None
        if time.monotonic() >= deadline:
            raise RuntimeError("Allocated GPU did not become idle; occupation samples are preserved")
        time.sleep(2)
        sample += 1


def verify_source(args, root, revision):
    local_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    local_changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            cwd=root, text=True)
    remote_revision = ssh(args, shlex.join(["git", "-C", args.worker_root, "rev-parse", "HEAD"]),
                          check=True, capture_output=True, text=True).stdout.strip()
    remote_changes = ssh(args, shlex.join(["git", "-C", args.worker_root, "status", "--porcelain",
                                         "--untracked-files=no"]), check=True, capture_output=True, text=True).stdout
    if local_revision != revision or revision != remote_revision or local_changes or remote_changes:
        raise RuntimeError("Unified acceptance requires matching clean fixed source checkouts")


@contextmanager
def stage_record(output, record, entry):
    try:
        yield
    finally:
        failure = sys.exception()
        if failure is not None:
            entry.update(state="aborted", error_type=type(failure).__name__, error=str(failure),
                         ended_at=datetime.now(timezone.utc).isoformat())
            save_json(output / "campaign.json", record)


def main():
    parser = argparse.ArgumentParser(description="串行执行单个 GPU 的原生验收，保留每个阶段的实际结果")
    parser.add_argument("--plan", type=Path, default=Path("configs/experiments/runtime-release-acceptance.json"))
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-root", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--worker-edh-source", required=True)
    parser.add_argument("--worker-policy-dir", required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--remote-provider-config", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--port", type=int, default=4365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch("[a-z][a-z0-9-]*", args.output.name) is None:
        raise ValueError("Campaign output directory name must use lowercase letters, digits and hyphens")
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    Draft202012Validator(STAGE_SCHEMA).validate(plan)
    if len({stage["id"] for stage in plan["stages"]}) != len(plan["stages"]):
        raise ValueError("Acceptance stage identities must be unique")
    if not 1 <= args.port <= 65535 or args.worker_host.startswith("-") or any(c.isspace() for c in args.worker_host):
        raise ValueError("Worker host or local port is invalid")
    for value in (args.worker_root, args.worker_python, args.worker_edh_source,
                  args.worker_policy_dir, args.remote_provider_config):
        if not value.startswith("/") or "\0" in value:
            raise ValueError("Remote paths must be absolute")
    for stage in plan["stages"]:
        if stage["kind"] == "metric":
            metrics = json.loads(within(root, stage["plan"]).read_text())
            Draft202012Validator(PLAN_SCHEMA).validate(metrics)
            if stage["suite"] not in metrics["suites"]:
                raise ValueError("Unknown metric suite")
            for case in metrics["suites"][stage["suite"]]["cases"]:
                case_motions(case)
        else:
            scene = json.loads(within(root, stage["scene"]).read_text())
            if scene["backend"] != "isaac-newton" or not within(root, stage["instruction"]).read_text().strip():
                raise ValueError("Navigation requires Newton and an actual instruction")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    verify_source(args, root, revision)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    remote_output = args.worker_root + "/outputs/release-campaigns/" + output.name
    record = {"schema_version": 1, "passed": False, "state": "running", "source_revision": revision,
              "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "physical_gpu": args.gpu,
              "worker_host": args.worker_host, "remote_output": remote_output, "stages": [],
              "scope": "Declared native runtime cases; broader generalization and hardware separate"}
    save_json(output / "plan.json", plan)
    save_json(output / "campaign.json", record)
    lock_path = root / ".cache/acceptance-gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with stage_record(output, record, record), lock_path.open("a") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_snapshot(args, output, "gpu-before")
        processes = ssh(args, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader",
                        check=True, capture_output=True, text=True).stdout
        (output / "gpu-processes-before.csv").write_text(processes)
        for stage in plan["stages"]:
            verify_source(args, root, revision)
            gpu_snapshot(args, output, stage["id"] + "-gpu-before")
            entry = {"id": stage["id"], "kind": stage["kind"], "state": "running",
                     "started_at": datetime.now(timezone.utc).isoformat()}
            record["stages"].append(entry)
            save_json(output / "campaign.json", record)
            started = time.monotonic()
            print(json.dumps(entry), flush=True)
            with stage_record(output, record, entry), (output / (stage["id"] + ".log")).open("x") as log:
                if stage["kind"] == "metric":
                    command = ["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", "ORT_DISABLE_TELEMETRY=1",
                        f"PYTHONPATH={args.worker_root}/src:{args.worker_edh_source}/harness/physical-runtime/src",
                        f"TMPDIR={args.worker_root}/.cache/tmp", args.worker_python,
                        "scripts/accept_metric_campaign.py", "--plan", stage["plan"], "--suite", stage["suite"],
                        "--catalog", args.worker_policy_dir, "--gpu", str(args.gpu),
                        "--output", remote_output + "/" + stage["id"]]
                    finished = ssh(args, "cd " + shlex.quote(args.worker_root) + " && " + shlex.join(command),
                                   stdout=log, stderr=subprocess.STDOUT)
                    subprocess.run(["rsync", "-a", args.worker_host + ":" + remote_output + "/" + stage["id"] + "/",
                                    str(output / stage["id"]) + "/"], check=True)
                else:
                    data = root / ".cache/harness-release" / output.name / stage["id"]
                    data.parent.mkdir(parents=True, exist_ok=True)
                    with socket.socket() as connection:
                        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
                            raise RuntimeError("Acceptance server port is already in use")
                    command = [sys.executable, str(root / "omd.py"), "harness", "--edh-source", str(args.edh_source),
                        "--remote-provider-config", args.remote_provider_config, "--ssh-host", args.worker_host,
                        "--remote-python", args.worker_python, "--model-api", "responses", "--reasoning-effort", "high",
                        "--worker-host", args.worker_host, "--worker-root", args.worker_root,
                        "--worker-python", args.worker_python, "--worker-edh-source", args.worker_edh_source,
                        "--worker-policy-dir", args.worker_policy_dir, "--worker-cuda-device", str(args.gpu),
                        "--scene-config", str(root / stage["scene"]), "--seed", "20261007",
                        "--port", str(args.port), "--data-dir", str(data)]
                    server = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    try:
                        deadline = time.monotonic() + 60
                        while True:
                            if server.poll() is not None:
                                raise RuntimeError("Native acceptance server exited during startup")
                            with socket.socket() as connection:
                                if connection.connect_ex(("127.0.0.1", args.port)) == 0:
                                    break
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Native acceptance server did not become ready")
                            time.sleep(0.5)
                        origin = f"http://127.0.0.1:{args.port}"
                        request(origin, "/api/config")
                        finished = subprocess.run([sys.executable, str(root / "scripts/run_navigation_acceptance.py"),
                            "--base-url", origin, "--scene-config", str(root / stage["scene"]),
                            "--instruction", str(root / stage["instruction"]), "--data-directory", str(data),
                            "--output", str(output / stage["id"]), "--minimum-distance-m", str(stage["minimum_distance_m"])],
                            cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    finally:
                        if server.poll() is None:
                            os.killpg(server.pid, signal.SIGTERM)
                        server.wait(timeout=30)
            entry.update(state="passed" if finished.returncode == 0 else "failed", exit_code=finished.returncode,
                         elapsed_s=time.monotonic() - started, ended_at=datetime.now(timezone.utc).isoformat())
            processes = ssh(args, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader",
                            check=True, capture_output=True, text=True).stdout
            (output / (stage["id"] + "-gpu-processes-after.csv")).write_text(processes)
            save_json(output / "campaign.json", record)
            print(json.dumps(entry), flush=True)
        record["passed"] = all(entry["state"] == "passed" for entry in record["stages"])
        record["state"] = "passed" if record["passed"] else "failed"
        processes = ssh(args, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader",
                        check=True, capture_output=True, text=True).stdout
        (output / "gpu-processes-after.csv").write_text(processes)
        save_json(output / "campaign.json", record)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
