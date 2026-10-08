import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from jsonschema import Draft202012Validator
from oh_my_duck.core.paths import project_root
from oh_my_duck.validation.metric.verify import verify_case
from oh_my_duck.validation.metric.policy import verify as verify_policy_inputs
from oh_my_duck.validation.metric.plans import PLAN_SCHEMA, case_motions
from oh_my_duck.infrastructure.owned_process import owned_process


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Metric campaign cancellation requested")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="Run sequential real policy-tool cases; stop on the first failure.")
    parser.add_argument("--plan", type=Path, default=Path("configs/experiments/metric-policy-acceptance.json"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    root = project_root()
    args.plan = args.plan if args.plan.is_absolute() else root / args.plan
    plan = json.loads(args.plan.read_text())
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    suite = plan["suites"][args.suite]
    cases = suite["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Metric campaign requires unique case identifiers")
    for case in cases:
        case_motions(case)
    scene = root / suite["scene_config"]
    configuration = json.loads(scene.read_text())
    backend = configuration["backend"]
    if backend not in {"cpu-mujoco-bam", "isaac-newton"}:
        raise ValueError("Metric campaign requires an actual supported simulation backend")
    if backend == "isaac-newton" and args.gpu not in {2, 3, 4}:
        raise ValueError("Newton campaign requires one allocated physical GPU from devices 2–4")
    if backend == "cpu-mujoco-bam" and args.gpu is not None:
        raise ValueError("CPU campaign does not allocate a GPU")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else ""
    environment["PYTHONPATH"] = os.pathsep.join([
        str(root / "src"),
        str(root / ".cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/harness/physical-runtime/src"),
    ])
    temporary = root / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"passed": False, "suite": args.suite, "backend": backend,
              "physical_gpu": args.gpu,
              "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "planned_cases": len(cases), "cases": []}
    destination_record = args.output / "campaign.json"
    destination_record.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for case in cases:
            destination = args.output / case["id"]
            entry = {"case": case, "passed": False, "state": "running",
                     "result": str(destination / "result.json")}
            record["cases"].append(entry)
            destination_record.write_text(json.dumps(record, indent=2) + "\n")
            sequence_path = args.output / (case["id"] + "-motions.json")
            sequence_path.write_text(json.dumps(case_motions(case), indent=2) + "\n")
            command = [sys.executable, "-m", "oh_my_duck.validation.metric.case",
                       "--scene-config", str(scene), "--catalog", str(args.catalog.resolve(strict=True)),
                       "--output", str(destination), "--seed", str(case["seed"]),
                       "--sequence", str(sequence_path)]
            print(f"Starting actual {args.suite}/{case['id']}", flush=True)
            with (args.output / (case["id"] + ".log")).open("w") as log:
                with owned_process(command, record_path=args.output / (case["id"] + "-process.json"),
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT) as worker:
                    finished = worker.wait()
                    if finished:
                        raise subprocess.CalledProcessError(finished, command)
            result_path = destination / "result.json"
            result = json.loads(result_path.read_text())
            if not result["passed"] or not result["resources_released"]:
                raise AssertionError("Metric case requires physical acceptance and resource release")
            verification = verify_case(destination, case)
            verification_path = destination / "verification.json"
            verification_path.write_text(json.dumps(verification, indent=2) + "\n")
            policy_verification = verify_policy_inputs(destination, args.catalog)
            policy_verification_path = destination / "policy-verification.json"
            policy_verification_path.write_text(json.dumps(policy_verification, indent=2) + "\n")
            entry.update(passed=True, state="passed", result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                         verification_sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest(),
                         policy_verification_sha256=hashlib.sha256(policy_verification_path.read_bytes()).hexdigest(),
                         measurements=result["measurements"])
            destination_record.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"case": case["id"], "passed": True,
                              "errors": [item["evidence"]["error"] for item in result["measurements"]]}), flush=True)
        record["passed"] = True
    finally:
        error = sys.exception()
        if error is not None:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
            if record["cases"] and not record["cases"][-1]["passed"]:
                record["cases"][-1]["state"] = "aborted" if isinstance(error, KeyboardInterrupt) else "failed"
        destination_record.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
