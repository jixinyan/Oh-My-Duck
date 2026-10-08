import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess

from oh_my_duck.core.paths import project_root


def verify(directory: Path, catalog: Path) -> dict:
    import numpy as np
    from PIL import Image

    from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES
    from oh_my_duck.validation.harness.pose import PHASES
    from oh_my_duck.validation.metric.policy import verify as verify_policy

    directory = directory.resolve(strict=True)
    record = json.loads((directory / "result.json").read_text())
    tools = json.loads((directory / "tools.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    frames = json.loads((directory / "frames.json").read_text())
    events = json.loads((directory / "events.json").read_text())
    provenance = record["provenance"]
    if provenance != json.loads((directory / "provenance.json").read_text()):
        raise AssertionError("Pose source metadata differs from its original record")
    if provenance["source_has_tracked_changes"] or provenance["source_diff_sha256"] != hashlib.sha256(b"").hexdigest():
        raise AssertionError("Pose campaign requires immutable clean source")
    source = subprocess.check_output(["git", "show", f"{provenance['source_revision']}:{provenance['runner_source_path']}"],
                                     cwd=project_root())
    if hashlib.sha256(source).hexdigest() != provenance["runner_sha256"]:
        raise AssertionError("Pose runner differs from its recorded source revision")
    if record["cuda_initialized"]:
        raise AssertionError("Pose campaign initialized CUDA")
    if record["provenance"]["configuration"]["backend"] != "cpu-mujoco-bam":
        raise AssertionError("Pose evidence requires actual CPU MuJoCo/BAM")
    if record["provenance"]["cuda_visible_devices"] != "":
        raise AssertionError("Pose evidence does not declare CPU-only execution")
    if not record["resources_released"] or not record["closed"]["closed"]:
        raise AssertionError("Pose campaign has no confirmed resource release")
    boundary = record["terminal_boundary"]
    if boundary["state"] != "ended" or not boundary["device_confirmed"]:
        raise AssertionError("Pose campaign has no confirmed native terminal boundary")
    execution = record["termination"]["execution"]
    if (boundary["executed_actions"] != 575 or boundary["reserved_actions"] != 575
            or execution["control_steps"] != 575 or execution["policy_calls"] != 575
            or execution["raw_sim_steps"] != 2300 or boundary["dispatch_in_flight"]
            or boundary["error"] is not None or execution["boundary_event_id"] != boundary["boundary_id"]):
        raise AssertionError("Pose terminal counters and physical execution identities differ")
    if record["termination"]["stop_confirmation"]["stopped_samples"] < 5:
        raise AssertionError("Pose campaign lacks measured physical stopping")
    policy = verify_policy(directory, catalog)
    if policy["policies"] != ["alpha_stand", "velstand"] or policy["control_steps"] != 575:
        raise AssertionError("Pose campaign differs from its complete official policy schedule")
    admitted = [sample for sample in samples if sample["control"] is not None and sample["control"]["executed_actions"]]
    if any(sample["control"]["raw_sim_steps"] != 4 for sample in admitted):
        raise AssertionError("Pose control did not execute four actual physics substeps")
    if any(sample["contact_evidence"]["non_ground_external_contact_samples_total"] for sample in admitted):
        raise AssertionError("Pose campaign contacted an external obstacle")
    means, controls = {}, []
    for name, head_pitch, body_height in PHASES:
        commands = [tool for tool in tools if tool["phase"] == name and tool["operation"] == "set_command"]
        if len(commands) != 1 or commands[0]["arguments"]["max_control_steps"] != 100:
            raise AssertionError("Pose phase lacks its single bounded native command")
        expected = np.zeros(13, dtype=np.float32)
        expected[4], expected[9] = head_pitch, body_height
        phase_samples = [sample for sample in admitted if sample["phase"] == name]
        if len(phase_samples) != 100:
            raise AssertionError("Pose phase did not execute all 100 native controls")
        admission = commands[0]["result"]
        if admission["command"] != {**commands[0]["arguments"]["command"], "posture": "stand"}:
            raise AssertionError("Pose native admission changed its requested command")
        if phase_samples[0]["sequence"] != admission["effective_after_sequence"] + 1:
            raise AssertionError("Pose command admission does not match the next physical action")
        for sample in phase_samples:
            inference = sample["control"]["policy_inference"]
            np.testing.assert_array_equal(np.asarray(inference["command_block"], dtype=np.float32), expected)
            if inference["policy"] != "alpha_stand":
                raise AssertionError("Pose phase executed another official policy")
            for key, shape in (("joint_position_rad", (14,)), ("joint_velocity_rad_s", (14,)),
                               ("body_position_m", (3,))):
                values = np.asarray(sample[key])
                if values.shape != shape or not np.isfinite(values).all():
                    raise AssertionError("Pose samples lack finite native servo and body measurements")
        joint_reads = [tool["result"] for tool in tools if tool["phase"] == name
                       and tool["operation"] == "read_sensor" and tool["arguments"]["sensor"] == "joint_state"]
        if len(joint_reads) != 1 or tuple(joint_reads[0]["measurements"]["joint_names"]) != JOINT_NAMES:
            raise AssertionError("Pose boundary lacks the actual named servo sensor")
        measured = joint_reads[0]
        if measured["sequence"] != phase_samples[-1]["sequence"]:
            raise AssertionError("Pose sensor does not belong to the completed phase")
        np.testing.assert_array_equal(measured["measurements"]["joint_position_rad"], phase_samples[-1]["joint_position_rad"])
        progress = [tool["result"] for tool in tools if tool["phase"] == name and tool["operation"] == "progress"][-1]
        if (progress["fallen"] or progress["height_m"] < 0.09 or progress["tilt_rad"] > np.deg2rad(25)
                or progress["sequence"] != phase_samples[-1]["sequence"]):
            raise AssertionError("Pose phase lacks its measured upright endpoint")
        head_reads = [tool["result"] for tool in tools if tool["phase"] == name
                      and tool["operation"] == "read_sensor" and tool["arguments"]["sensor"] == "head_rgb"]
        if len(head_reads) != 1 or head_reads[0]["sequence"] != measured["sequence"]:
            raise AssertionError("Pose head camera does not belong to the actual boundary")
        with Image.open(BytesIO(base64.b64decode(head_reads[0]["measurements"]["rgb_png_base64"], validate=True))) as image:
            image.load()
            if image.format != "PNG" or image.size != (320, 240) or np.ptp(np.asarray(image.convert("RGB"))) == 0:
                raise AssertionError("Pose head camera has invalid original RGB pixels")
        window = phase_samples[-25:]
        means[name] = {"head_pitch_rad": float(np.mean([sample["joint_position_rad"][6] for sample in window])),
                       "body_height_m": float(np.mean([sample["body_position_m"][2] for sample in window]))}
        controls.append({"phase": name, "controls": len(phase_samples), "command_block": expected.tolist(),
                         "first_sequence": phase_samples[0]["sequence"], "last_sequence": phase_samples[-1]["sequence"]})
    responses = {}
    for name, baseline, returned, direction in (("positive", "neutral", "return_positive", 1),
                                                ("negative", "return_positive", "return_negative", -1)):
        response = {}
        for key, threshold in (("head_pitch_rad", 0.03), ("body_height_m", 0.001)):
            change = means[name][key] - means[baseline][key]
            restored_error = abs(means[returned][key] - means[baseline][key])
            if change * direction < threshold or restored_error >= abs(change):
                raise AssertionError(f"Measured pose response failed for {name}/{key}: {change}, {restored_error}")
            response[key] = {"change": change, "minimum_directional_change": threshold,
                             "return_error": restored_error}
        responses[name] = response
    if {frame["phase"] for frame in frames} != {"warmup", *(item[0] for item in PHASES)}:
        raise AssertionError("Pose recording omits an actual phase camera")
    native_frames = [event["data"] for event in events if event["event"] == "frame"]
    if len(native_frames) != len(frames):
        raise AssertionError("Pose saved frame count differs from actual native events")
    for frame, native in zip(frames, native_frames, strict=True):
        path = (directory / frame["file"]).resolve(strict=True)
        if not path.is_relative_to(directory / "frames"):
            raise ValueError("Pose frame path must remain within its frames directory")
        original = base64.b64decode(native["observation"]["images"]["observer_follow"], validate=True)
        if path.read_bytes() != original or frame["simulation_time_s"] != native["simulation_time_s"]:
            raise AssertionError("Pose camera bytes or simulation time differ from the actual native event")
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image.convert("RGB"))
            if image.format != "PNG" or pixels.shape != (480, 640, 3) or np.ptp(pixels) == 0:
                raise AssertionError("Pose observer camera does not contain an actual valid RGB frame")
    return {"passed": True, "scope": "Native command admission, actual CPU head pitch/body height response and return",
            "control_steps": len(admitted), "raw_sim_steps": 4 * len(admitted), "phase_controls": controls,
            "phase_means": means, "responses": responses, "camera_frames": len(frames), "policy_audit": policy,
            "source_revision": record["provenance"]["source_revision"],
            "input_sha256": {f"{name}.json": hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest()
                             for name in ("samples", "events", "tools", "frames")},
            "gpu_acceptance_performed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory, args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
