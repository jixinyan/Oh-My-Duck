import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from oh_my_duck.infrastructure.owned_process import owned_process


def main():
    parser = argparse.ArgumentParser(description="使用实际 CPU policy 运动检查控制连接中断与进程清理")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("control-eof", "sigterm"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", ORT_DISABLE_TELEMETRY="1")
    environment["PYTHONPATH"] = os.pathsep.join([
        str(root / "src"),
        str(root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/harness/physical-runtime/src"),
    ])
    environment["TMPDIR"] = str(root / ".cache/tmp")
    case_output = output / "campaign"
    command = [sys.executable, str(root / "scripts/run_owned_acceptance.py"),
               "--record", str(output / "supervisor.json"), "--", sys.executable,
               str(root / "scripts/accept_metric_campaign.py"), "--plan",
               "configs/experiments/metric-policy-acceptance.json", "--suite", "apartment-feet",
               "--catalog", str(args.catalog.resolve(strict=True)), "--output", str(case_output)]
    with (output / "controller.log").open("x") as log:
        with owned_process(command, record_path=output / "controller.json", cancel_stdin=True,
                           stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                           cwd=root, env=environment) as supervisor:
            deadline = time.monotonic() + 120
            while True:
                if supervisor.poll() is not None:
                    raise RuntimeError("Actual CPU campaign exited before the requested interruption")
                log_path = case_output / "forward-turn.log"
                if log_path.exists() and '"operation": "walk"' in log_path.read_text():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Actual CPU campaign did not start its metric motion")
                time.sleep(0.1)
            time.sleep(0.2)
            if args.mode == "control-eof":
                supervisor.stdin.close()
            else:
                os.kill(supervisor.pid, signal.SIGTERM)
            exit_code = supervisor.wait(timeout=150)
    if exit_code != 130:
        raise AssertionError("Interrupted supervisor must preserve cancellation exit status")
    campaign = json.loads((case_output / "campaign.json").read_text())
    result = json.loads((case_output / "forward-turn/result.json").read_text())
    samples = json.loads((case_output / "forward-turn/samples.json").read_text())
    worker = json.loads((case_output / "forward-turn-process.json").read_text())
    supervisor = json.loads((output / "supervisor.json").read_text())
    if (campaign["passed"] or result["passed"] or not result["closed"]["closed"] or
            not result["resources_released"] or campaign["cases"][-1]["state"] != "aborted"):
        raise AssertionError("Interrupted actual campaign did not preserve failure and native resource release")
    if any(item.get("grace_timeout") or item["state"] != "exited" for item in (worker, supervisor)):
        raise AssertionError("Actual cancellation exceeded graceful process cleanup")
    controls = max(sample["sequence"] for sample in samples)
    if controls <= 75 or any(sample["contact_evidence"]["non_ground_external_contact_samples_total"]
                            for sample in samples):
        raise AssertionError("Interruption requires actual policy movement and zero external contact")
    report = {"passed": True, "scope": "Actual CPU policy campaign cancellation and resource release",
              "mode": args.mode, "exit_code": exit_code, "control_steps": controls,
              "resources_released": result["resources_released"],
              "supervisor": supervisor, "worker": worker}
    (output / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
