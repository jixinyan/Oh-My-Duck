import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shlex
import re
import signal
import socket
import subprocess
import sys
import tarfile
import time

from oh_my_duck.core.paths import project_root
from oh_my_duck.experience.harness_replay import request, save_json
from oh_my_duck.validation.release.plans import validate_plan
from oh_my_duck.infrastructure.owned_process import owned_process


def ssh(args, command, **kwargs):
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                           args.worker_host, command], **kwargs)


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


def preflight(args, root, output, remote_output, revision):
    local_pin = subprocess.check_output(["git", "-C", str(args.edh_source), "rev-parse", "HEAD"], text=True).strip()
    if local_pin != "8a5e685b22d032207f53db20454f0992a4ad60fd":
        raise ValueError("Local native Harness differs from its pinned source")
    (args.edh_source / "node_modules/tsx/dist/loader.mjs").resolve(strict=True)
    archive = subprocess.check_output(["git", "-C", str(args.edh_source), "archive", local_pin,
        "harness", "apps/server", "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json"])
    expected = {}
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            if member.isfile():
                expected[member.name] = hashlib.sha256(source.extractfile(member).read()).hexdigest()
    for name, digest in expected.items():
        if hashlib.sha256((args.edh_source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Local Harness runtime source differs from its pinned Git content")
    worker_files = {name: digest for name, digest in expected.items()
                    if (name.startswith("harness/physical-runtime/src/") and name.endswith(".py")) or
                    name == "harness/contracts/schema/physical.schema.json"}
    manifest = output / "harness-source.json"
    save_json(manifest, {"revision": local_pin, "role": "remote_physical_worker", "files": worker_files,
                        "local_runtime_files_verified": len(expected)})
    ssh(args, "mkdir -p " + shlex.quote(args.worker_root + "/outputs/release-campaigns") +
        " && mkdir " + shlex.quote(remote_output), check=True)
    subprocess.run(["rsync", "-a", str(manifest), args.worker_host + ":" + remote_output + "/harness-source.json"], check=True)
    subprocess.run(["node", "--check", str(root / "integrations/edh/server.mjs")], check=True)
    command = ["env", "CUDA_VISIBLE_DEVICES=", "ORT_DISABLE_TELEMETRY=1",
               f"PYTHONPATH={args.worker_root}/src", f"TMPDIR={args.worker_root}/.cache/tmp",
               args.worker_python, "scripts/preflight_release_worker.py", "--plan",
               str(args.plan.resolve().relative_to(root)), "--catalog", args.worker_policy_dir,
               "--edh-source", args.worker_edh_source, "--harness-manifest", remote_output + "/harness-source.json",
               "--provider-config", args.remote_provider_config,
               "--output", remote_output + "/preflight"]
    with (output / "preflight.log").open("x") as log:
        try:
            ssh(args, "cd " + shlex.quote(args.worker_root) + " && " + shlex.join(command),
                check=True, stdout=log, stderr=subprocess.STDOUT)
        finally:
            subprocess.run(["rsync", "-a", args.worker_host + ":" + remote_output + "/preflight/",
                            str(output / "preflight") + "/"], check=True)
    actual = json.loads((output / "preflight/result.json").read_text())
    if not actual["passed"] or actual["source_revision"] != revision or actual["cuda_runtime_checked"]:
        raise RuntimeError("Remote metadata preflight failed its source and readiness checks")
    return {"passed": True, "local_harness_revision": local_pin,
            "remote_report_sha256": hashlib.sha256((output / "preflight/result.json").read_bytes()).hexdigest(),
            "cuda_runtime_checked": False}


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
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Release campaign cancellation requested")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="串行执行单个 GPU 的原生验收，保留每个阶段的实际结果")
    parser.add_argument("--plan", type=Path, default=Path("configs/experiments/runtime-release-acceptance.json"))
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-root", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--worker-edh-source", required=True)
    parser.add_argument("--worker-policy-dir", required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--remote-provider-config", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3, 4))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--port", type=int, default=4365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.preflight_only and args.gpu is None:
        raise ValueError("GPU acceptance requires one allocated physical device from 2–4")
    if re.fullmatch("[a-z][a-z0-9-]*", args.output.name) is None:
        raise ValueError("Campaign output directory name must use lowercase letters, digits and hyphens")
    root = project_root()
    args.plan = args.plan if args.plan.is_absolute() else root / args.plan
    plan, _scenes, inputs = validate_plan(root, args.plan)
    if not 1 <= args.port <= 65535 or args.worker_host.startswith("-") or any(c.isspace() for c in args.worker_host):
        raise ValueError("Worker host or local port is invalid")
    for value in (args.worker_root, args.worker_python, args.worker_edh_source,
                  args.worker_policy_dir, args.remote_provider_config):
        if not value.startswith("/") or "\0" in value:
            raise ValueError("Remote paths must be absolute")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    verify_source(args, root, revision)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    remote_output = args.worker_root + "/outputs/release-campaigns/" + output.name
    record = {"schema_version": 1, "passed": False, "state": "running", "source_revision": revision,
              "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "physical_gpu": args.gpu,
              "input_sha256": inputs, "gpu_acceptance_performed": False,
              "worker_host": args.worker_host, "remote_output": remote_output, "stages": [],
              "scope": "Declared native runtime cases; broader generalization and hardware separate"}
    save_json(output / "plan.json", plan)
    save_json(output / "campaign.json", record)
    lock_path = root / ".cache/acceptance-gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with stage_record(output, record, record), lock_path.open("a") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record["preflight"] = preflight(args, root, output, remote_output, revision)
        if args.preflight_only:
            record["state"] = "preflight_passed"
            save_json(output / "campaign.json", record)
            print(json.dumps({"state": "preflight_passed", "gpu_acceptance_performed": False}), flush=True)
            return 0
        save_json(output / "campaign.json", record)
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
            record["gpu_acceptance_performed"] = True
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
                    environment = command[:5]
                    lifecycle = remote_output + "/" + stage["id"] + "-process.json"
                    supervised = environment + [args.worker_python, "scripts/run_owned_acceptance.py",
                                                "--record", lifecycle, "--"] + command[5:]
                    remote_command = "cd " + shlex.quote(args.worker_root) + " && " + shlex.join(supervised)
                    try:
                        with owned_process(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                                            args.worker_host, remote_command],
                                           record_path=output / (stage["id"] + "-transport.json"),
                                           cancel_stdin=True, stdin=subprocess.PIPE,
                                           stdout=log, stderr=subprocess.STDOUT) as transport:
                            exit_code = transport.wait()
                    finally:
                        subprocess.run(["rsync", "-a", "--include=" + stage["id"] + "/***",
                                        "--include=" + stage["id"] + "-process.json", "--exclude=*",
                                        args.worker_host + ":" + remote_output + "/", str(output) + "/"], check=True)
                    cleanup = json.loads((output / (stage["id"] + "-process.json")).read_text())
                    if cleanup["state"] != "exited" or cleanup.get("grace_timeout"):
                        raise RuntimeError("Remote metric supervisor lacks graceful exit evidence")
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
                    with owned_process(command, record_path=output / (stage["id"] + "-server.json"),
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT) as server:
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
                        with owned_process([sys.executable, str(root / "scripts/run_navigation_acceptance.py"),
                            "--base-url", origin, "--scene-config", str(root / stage["scene"]),
                            "--instruction", str(root / stage["instruction"]), "--data-directory", str(data),
                            "--output", str(output / stage["id"]), "--minimum-distance-m", str(stage["minimum_distance_m"])],
                            record_path=output / (stage["id"] + "-process.json"),
                            cwd=root, stdout=log, stderr=subprocess.STDOUT) as navigation:
                            exit_code = navigation.wait()
            entry.update(state="passed" if exit_code == 0 else "failed", exit_code=exit_code,
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
