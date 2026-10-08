import argparse
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageStat
from metric_plan import case_motions


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Artifact contains a non-finite number")


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"Physical measurement mismatch: {actual} != {expected}")


def verify_case(directory, expected):
    result = json.loads((directory / "result.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    tools = json.loads((directory / "tools.json").read_text())
    finite(result)
    finite(samples)
    require(result["passed"] and result["resources_released"] and result["closed"]["closed"],
            "Case requires successful motion and session cleanup")
    provenance = result["provenance"]
    require(provenance == json.loads((directory / "provenance.json").read_text()),
            "Saved provenance differs from the case result")
    require(not provenance["source_has_tracked_changes"], "Acceptance source has tracked changes")
    require(provenance["source_diff_sha256"] == hashlib.sha256(b"").hexdigest(),
            "Acceptance source difference is not empty")
    require(provenance["configuration"]["seed"] == expected["seed"], "Case seed mismatch")
    motions = case_motions(expected)
    operations = [motion["operation"] for motion in motions]
    require(provenance["commands"]["operations"] == operations, "Case operation mismatch")
    if "motions" in provenance["commands"]:
        require(provenance["commands"]["motions"] == motions, "Case motion parameters differ")
    indexed = {}
    episode = samples[0]["episode_id"]
    for sample in samples:
        require(sample["episode_id"] == episode, "Case changed the physical episode")
        require(sample["contact_evidence"]["non_ground_external_contact_samples_total"] == 0,
                "Physical samples contain external obstacle contact")
        require(len(sample["tof_distance_mm"]) == 64 and
                all(0 <= distance <= 4000 for distance in sample["tof_distance_mm"]),
                "Physical ToF sample is outside its declared range")
        sequence = sample["sequence"]
        if sequence in indexed:
            for key in ("body_position_m", "yaw_rad", "body_twist", "contact_evidence"):
                require(sample[key] == indexed[sequence][key], "Repeated sequence changed physical state")
        else:
            require(sequence == len(indexed), "Physical control sample sequence is incomplete")
            indexed[sequence] = sample
        if sample["control"] is not None:
            control = sample["control"]
            require(len(control["action"]) == 14 and control["executed_actions"] == 1 and
                    control["raw_sim_steps"] == 4 and control["action_completed"],
                    "Policy action violates servo count or physical cadence")
    boundary = result["terminal_boundary"]
    execution = result["termination"]["execution"]
    count = len(indexed) - 1
    for tool in tools:
        if tool["operation"] == "observe":
            observed = tool["result"]
            measured = observed["measurements"]
            require(observed["episode_id"] == episode and observed["sequence"] in indexed and
                    observed["body_position_m"] == indexed[observed["sequence"]]["body_position_m"],
                    "Combined observation differs from original physics")
            require(len(set(measured["joint_names"])) == len(measured["joint_position_rad"]) ==
                    len(measured["joint_velocity_rad_s"]) == 14 and len(measured["tof_distance_mm"]) == 64,
                    "Combined observation has incomplete sensors")
            require(all(0 <= value <= 4000 for value in measured["tof_distance_mm"]), "Observed ToF is out of range")
            require(observed["execution"]["state"] in {"paused", "ended"} and
                    observed["execution"]["device_confirmed"] and
                    observed["execution"]["remaining_actions"] >= 0 and
                    observed["execution"]["remaining_wall_time_s"] >= 0,
                    "Combined observation lacks a confirmed native boundary or budget")
            with Image.open(BytesIO(base64.b64decode(observed["rgb_png_base64"], validate=True))) as image:
                require(image.size == (320, 240) and max(ImageStat.Stat(image.convert("RGB")).stddev) > 1,
                        "Combined observation has invalid head camera pixels")
    require(boundary["state"] == "ended" and boundary["device_confirmed"] and
            not boundary["dispatch_in_flight"] and boundary["error"] is None,
            "Native execution lacks a confirmed terminal boundary")
    require(execution["state"] == "ended" and execution["device_confirmed"] and
            execution["boundary_event_id"] == boundary["boundary_id"], "Terminal identities differ")
    require(count == boundary["executed_actions"] == boundary["reserved_actions"] ==
            execution["control_steps"] == execution["policy_calls"] and execution["raw_sim_steps"] == 4 * count,
            "Execution counters disagree with physical samples")
    measurements = result["measurements"]
    require([item["operation"] for item in measurements] == operations,
            "Measured operations differ from the declared case")
    errors = []
    for item, motion in zip(measurements, motions, strict=True):
        evidence, progress = item["evidence"], item["progress"]
        operation = item["operation"]
        key = "distance_m" if operation == "walk" else "angle_deg"
        speed_key = "speed_m_s" if operation == "walk" else "angular_speed_deg_s"
        parameters = motion["arguments"]
        close(item["arguments"][key], parameters[key])
        close(item["arguments"][speed_key], parameters[speed_key])
        close(evidence["requested"], parameters[key])
        admissions = [tool for tool in tools if tool["operation"] == operation and
                      tool["result"].get("request_id") == evidence["request_id"]]
        require(len(admissions) == 1, "Measured operation has no unique native tool admission")
        admission = admissions[0]["result"]
        start_sequence = admission["command_admission"]["effective_after_sequence"]
        end_sequence = evidence["sequence"]
        require(0 <= start_sequence < end_sequence <= count, "Motion sequence range is invalid")
        first, last = indexed[start_sequence], indexed[end_sequence]
        require(first["body_position_m"] == evidence["start_position_m"] == admission["start_position_m"] and
                last["body_position_m"] == evidence["body_position_m"] == progress["body_position_m"],
                "Measured endpoints disagree with physical samples")
        close(first["yaw_rad"], evidence["start_yaw_rad"])
        close(last["yaw_rad"], evidence["yaw_rad"])
        dx = last["body_position_m"][0] - first["body_position_m"][0]
        dy = last["body_position_m"][1] - first["body_position_m"][1]
        close(evidence["translation_xy_m"], math.hypot(dx, dy))
        if operation == "walk":
            yaw = first["yaw_rad"]
            measured = dx * math.cos(yaw) + dy * math.sin(yaw)
            error = math.hypot(dx - parameters[key] * math.cos(yaw), dy - parameters[key] * math.sin(yaw))
            tolerance = 0.05
        else:
            turn = 0.0
            for sequence in range(start_sequence + 1, end_sequence + 1):
                delta = indexed[sequence]["yaw_rad"] - indexed[sequence - 1]["yaw_rad"]
                turn += math.atan2(math.sin(delta), math.cos(delta))
            measured = math.degrees(turn)
            error = abs(parameters[key] - measured)
            tolerance = 5.0
        close(evidence["measured"], measured)
        close(evidence["error"], error)
        close(evidence["tolerance"], tolerance)
        require(error <= tolerance and evidence["completed"] and evidence["phase"] == "complete",
                "Measured motion does not meet the declared physical tolerance")
        require(progress["sequence"] == end_sequence and not progress["fallen"] and
                progress["height_m"] >= 0.09 and progress["tilt_rad"] <= math.radians(25),
                "Motion endpoint is not upright")
        require(progress["stopped_samples"] >= 5 and evidence["stopped_samples"] >= 5 and
                progress["command_block"][:3] == [0, 0, 0], "Motion endpoint lacks a zero-command stop")
        close(progress["simulation_time_s"], end_sequence / 50)
        for sequence in range(end_sequence - 4, end_sequence + 1):
            sample = indexed[sequence]
            twist = sample["body_twist"]
            require(math.hypot(twist[0], twist[1]) <= 0.025 and abs(twist[2]) <= 0.08 and
                    sample["body_position_m"][2] >= 0.09,
                    "Endpoint lacks five consecutive physical stopped samples")
        errors.append(error)
    frames = json.loads((directory / "frames.json").read_text())
    events = json.loads((directory / "events.json").read_text())
    native_frames = [event["data"] for event in events if event["event"] == "frame"]
    require(bool(frames), "Case has no observer camera frames")
    require(len(frames) == len(native_frames), "Saved camera frame count differs from native events")
    files = [directory / name for name in ("result.json", "provenance.json", "samples.json", "tools.json",
                                         "events.json", "frames.json", "head-final.png")]
    for frame, native in zip(frames, native_frames, strict=True):
        path = (directory / frame["file"]).resolve(strict=True)
        require(path.is_relative_to(directory.resolve()), "Camera frame path escapes the case directory")
        data = path.read_bytes()
        require(data == base64.b64decode(native["observation"]["images"]["observer_follow"], validate=True),
                "Observer image differs from its native physical frame")
        close(frame["simulation_time_s"], native["simulation_time_s"])
        require(native["run_task_id"] == result["task_id"] and
                native["execution_id"] == boundary["execution_id"], "Camera execution identity mismatch")
        with Image.open(BytesIO(data)) as image:
            image.verify()
        files.append(path)
    with Image.open(BytesIO((directory / "head-final.png").read_bytes())) as image:
        image.load()
        require(image.size == (320, 240), "Head camera dimensions changed")
        statistics = ImageStat.Stat(image.convert("RGB"))
        require(max(statistics.stddev) > 1, "Head camera has no measurable image variation")
    return {"passed": True, "case": expected["id"], "errors": errors, "control_steps": count,
            "physics_steps": execution["raw_sim_steps"], "source_revision": provenance["source_revision"],
            "backend": provenance["configuration"]["backend"],
            "cuda_visible_devices": provenance["cuda_visible_devices"],
            "artifact_sha256": {str(path.relative_to(directory)): digest(path) for path in files}}


def verify_campaign(directory, plan_path, source_root):
    campaign_path = directory / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    require(campaign["passed"] and len(campaign["cases"]) == campaign["planned_cases"],
            "Campaign is incomplete or failed")
    require(digest(plan_path) == campaign["plan_sha256"], "Campaign plan hash changed")
    suite = json.loads(plan_path.read_text())["suites"][campaign["suite"]]
    require([entry["case"] for entry in campaign["cases"]] == suite["cases"],
            "Campaign cases differ from the complete declared plan")
    reports = []
    for entry in campaign["cases"]:
        require(entry["passed"] and entry["state"] == "passed", "Campaign case did not pass")
        case = entry["case"]
        child = (directory / case["id"]).resolve(strict=True)
        require(child.parent == directory.resolve(), "Case path escapes the campaign directory")
        require(digest(child / "result.json") == entry["result_sha256"], "Case result hash changed")
        report = verify_case(child, case)
        provenance = json.loads((child / "provenance.json").read_text())
        revision = provenance["source_revision"]
        require(re.fullmatch("[0-9a-f]{40}", revision) is not None, "Source revision is invalid")
        for name, key in (("scripts/accept_metric_policy_tools.py", "runner_sha256"),
                          (suite["scene_config"], "scene_config_sha256")):
            original = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=source_root)
            require(hashlib.sha256(original).hexdigest() == provenance[key],
                    "Runner or scene hash differs from the recorded source revision")
        require(report["backend"] == campaign["backend"], "Campaign backend differs from the case")
        expected_device = "" if campaign["physical_gpu"] is None else str(campaign["physical_gpu"])
        require(report["cuda_visible_devices"] == expected_device, "Campaign physical GPU allocation changed")
        reports.append(report)
    require(len({report["source_revision"] for report in reports}) == 1,
            "Campaign cases used different source revisions")
    return {"passed": True, "scope": "Independent saved physical artifact verification; model task acceptance separate",
            "campaign_sha256": digest(campaign_path), "cases": reports}


def main():
    parser = argparse.ArgumentParser(description="Recompute metric acceptance from saved physical samples.")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--plan", type=Path, default=root / "configs/experiments/metric-policy-acceptance.json")
    parser.add_argument("--source-root", type=Path, default=root)
    args = parser.parse_args()
    report = verify_campaign(args.campaign.resolve(strict=True), args.plan, args.source_root)
    with args.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
