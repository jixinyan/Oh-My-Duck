import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import time

from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION, POLICY_SHA256


def run(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    policy = root / ".cache" / "official-policies" / OFFICIAL_REVISION / "alpha_walking.onnx"
    if not policy.is_file():
        raise FileNotFoundError(policy)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(root / ".cache" / "tmp")
    environment["PYTHONPATH"] = str(root / "src")
    with (output / "service-stderr.log").open("w") as stderr:
        process = subprocess.Popen(
            [str(root / ".cache" / "cpu-apartment-locked-venv" / "bin" / "python"),
             str(root / "omd.py"), "sim", "--backend", "cpu-mujoco-bam", "--policy", str(policy)],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("JSONL service pipes were unavailable")
        responses = []

        def request(tool: str, request_id: str, arguments: dict) -> dict:
            process.stdin.write(json.dumps({"tool": tool, "request_id": request_id,
                                            "arguments": arguments}) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], 30.0)
            if not ready:
                raise TimeoutError(f"JSONL tool response timed out: {tool}:{request_id}")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"JSONL service closed before {tool}:{request_id}")
            result = json.loads(line)
            if result["request_id"] != request_id:
                raise ValueError(f"JSONL response ID mismatch: {tool}:{request_id}")
            responses.append({"tool": tool, "response": result})
            (output / "response-trace.json").write_text(json.dumps(responses, indent=2) + "\n")
            return result

        capability = request("get_capabilities", "capability", {})
        if capability["status"] != "ok" or "move_for" not in capability["payload"]["skills"]:
            raise ValueError("CPU simulator did not advertise the executable move_for skill")
        time.sleep(0.2)
        initial = request("get_robot_state", "initial", {})
        if initial["status"] != "ok" or initial["payload"]["sequence"] < 1:
            raise ValueError("CPU simulator did not produce a physical state")
        imu = request("read_sensor", "imu", {"sensor_id": "imu", "max_age_ms": 1000})
        joints = request("read_sensor", "joints", {"sensor_id": "joint_state", "max_age_ms": 1000})
        if imu["payload"]["validity"] != "valid" or joints["payload"]["validity"] != "valid":
            raise ValueError("Native IMU or joint sensor was stale")
        if len(joints["payload"]["readings"]["joint_names"]) != 14:
            raise ValueError("Joint sensor omitted physical servo channels")
        start = request("run_skill", "run-short", {"skill_id": "move_for", "parameters": {
            "vx_m_s": 0.3, "vy_m_s": 0.0, "yaw_rad_s": 0.0, "duration_s": 0.4}})
        if start["status"] != "accepted":
            raise ValueError("Short physical movement was not admitted")
        task_id = start["payload"]["task_id"]
        finished = None
        for poll in range(50):
            time.sleep(0.1)
            finished = request("get_task_status", f"status-short:{poll}", {"task_id": task_id})
            if finished["payload"]["status"] in {"succeeded", "failed", "cancelled"}:
                break
        if finished["status"] != "ok" or finished["payload"]["status"] not in {"succeeded", "failed"}:
            raise ValueError("Short physical movement did not reach a terminal status")
        after_move = request("get_robot_state", "after-move", {})
        start_long = request("run_skill", "run-cancel", {"skill_id": "move_for", "parameters": {
            "vx_m_s": 0.3, "vy_m_s": 0.0, "yaw_rad_s": 0.0, "duration_s": 2.0}})
        if start_long["status"] != "accepted":
            raise ValueError("Cancellable physical movement was not admitted")
        cancel = request("cancel_task", "cancel", {"task_id": start_long["payload"]["task_id"],
                                                   "reason": "acceptance_cancel"})
        if cancel["status"] != "ok" or cancel["payload"]["status"] != "cancelled":
            raise ValueError("Physical cancellation was not confirmed")
        cancelled_status = request("get_task_status", "status-cancelled",
                                   {"task_id": start_long["payload"]["task_id"]})
        if cancelled_status["payload"]["status"] != "cancelled":
            raise ValueError("Cancelled task status changed")
        stop = request("stop_motion", "stop", {})
        if stop["status"] != "ok" or not stop["payload"]["confirmed_stopped"]:
            raise ValueError("Physical stop was not confirmed")
        final = request("get_robot_state", "final", {})
        if final["payload"]["sequence"] <= initial["payload"]["sequence"]:
            raise ValueError("MuJoCo control sequence did not advance")
        if final["payload"]["measurements"]["policy_action"] == initial["payload"]["measurements"]["policy_action"]:
            raise ValueError("Official policy action did not respond to the motion command")
        process.stdin.close()
        exit_code = process.wait(timeout=15)
        if exit_code != 0:
            raise RuntimeError(f"JSONL simulator exited with code {exit_code}")
        return {"status": "passed", "policy": "alpha_walking",
                "policy_sha256": POLICY_SHA256["alpha_walking.onnx"],
                "initial_sequence": initial["payload"]["sequence"],
                "final_sequence": final["payload"]["sequence"],
                "initial_position_m": initial["payload"]["measurements"]["body_position_m"],
                "final_position_m": final["payload"]["measurements"]["body_position_m"],
                "short_task_status": finished["payload"]["status"],
                "short_task_details": finished["payload"].get("details"),
                "cancel_status": cancel["payload"]["status"],
                "stop_confirmed": stop["payload"]["confirmed_stopped"],
                "service_exit_code": exit_code, "responses": responses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "responses"}, indent=2))


if __name__ == "__main__":
    main()
