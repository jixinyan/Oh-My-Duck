import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path


def verify(capture: Path, catalog: Path | None = None) -> dict:
    import numpy as np
    from PIL import Image

    from oh_my_duck.perception.frames import bbox_mask, decode_frame
    from oh_my_duck.perception.apartment_objects import APARTMENT_OBJECTS
    from oh_my_duck.perception.tracking import TargetTracker
    from oh_my_duck.perception.validation import validate_measurements, validate_response
    from oh_my_duck.validation.metric.policy import verify as verify_policy

    totals = {"control_steps": 0, "raw_sim_steps": 0,
              "captures": 0, "native_pixel_identities": 0, "model_confirmed_observations": 0,
              "repeated_frames": 0, "observer_frames": 0, "invalid_measurements_rejected": 0}
    model_instances = {}
    model_sequences = set()
    policy_audits = []
    digests = {}
    for source in ("simulator_ground_truth", "models"):
        directory = capture / source
        result = json.loads((directory / "result.json").read_text())
        records = json.loads((directory / "captures.json").read_text())
        tools = json.loads((directory / "tools.json").read_text())
        samples = json.loads((directory / "samples.json").read_text())
        if catalog is not None:
            policy_audits.append(verify_policy(directory, catalog))
        tracker = TargetTracker()
        history = {}
        if not result["passed"] or result["cuda_initialized"] or not result["resources_closed"]:
            raise AssertionError("Tracking requires completed CPU sessions and released resources")
        controls = {item["sequence"]: item for item in samples
                    if item["control"] is not None and item["control"]["executed_actions"] > 0}
        if len(controls) != result["control_steps"] or result["raw_sim_steps"] != result["control_steps"] * 4:
            raise AssertionError("Tracking requires original 50 Hz / four-substep policy evidence")
        if sorted(controls) != list(range(1, result["control_steps"] + 1)):
            raise AssertionError("Tracking control samples contain gaps")
        states = {}
        for sample in samples:
            sequence = sample["sequence"]
            if type(sequence) is not int or not 0 <= sequence <= result["control_steps"]:
                raise AssertionError("Tracking update has an invalid physical sequence")
            physical = {key: value for key, value in sample.items() if key not in ("control", "phase")}
            if sequence in states and states[sequence] != physical:
                raise AssertionError("Repeated tracking update changed its physical measurement")
            states[sequence] = physical
        if catalog is not None and policy_audits[-1]["control_steps"] != len(controls):
            raise AssertionError("Tracking controls differ from independent ONNX evidence")
        totals["control_steps"] += result["control_steps"]
        totals["raw_sim_steps"] += result["raw_sim_steps"]
        for row in records:
            perception = row["perception"]
            if perception["sequence"] != row["physical_sequence"]:
                raise AssertionError("Tracking perception differs from its physical capture")
            replay = deepcopy(perception)
            tracker.update(replay, source)
            if replay != perception:
                raise AssertionError("Recorded tracking differs from original measured association")
            with np.load(directory / row["geometry"], allow_pickle=False) as geometry:
                segmentation = geometry["segmentation"]
            if source == "models":
                if row["device"] != "cpu":
                    raise AssertionError("Model tracking evidence requires CPU inference")
                frame = json.loads((directory / row["frame"]).read_text())
                response = json.loads((directory / row["response"]).read_text())
                validate_response(frame, perception["prompt"], response)
                validate_measurements(frame, response)
                raw = deepcopy(perception)
                raw.pop("tracking")
                for target in raw["targets"]:
                    target.pop("track_id")
                    target.pop("tracking")
                if raw != response:
                    raise AssertionError("Native tracking changed its original model response")
                image = decode_frame(frame).image
                if segmentation.shape != (image.height, image.width, 2):
                    raise AssertionError("Native instance pixels differ from model image dimensions")
            else:
                frame = json.loads((directory / row["physical_frame"]).read_text())
                original = decode_frame(frame)
                for target in perception["targets"]:
                    geom_ids = [index for index, name in enumerate(row["geom_names"])
                                if name == target["target_id"] or name.startswith(target["target_id"] + "_")]
                    mask = np.isin(segmentation[..., 0], geom_ids) & (segmentation[..., 1] == 5)
                    pixels = original.points_world_m[mask & np.isfinite(original.points_world_m).all(axis=-1)]
                    if target["valid_depth_pixels"] != len(pixels):
                        raise AssertionError("Native track pixels differ from original depth")
                    if len(pixels) >= 8:
                        np.testing.assert_allclose(target["surface_position_world_m"], np.median(pixels, axis=0),
                                                   rtol=0, atol=1e-6)
            for target in perception["targets"]:
                identifier = target["track_id"]
                if identifier is None:
                    continue
                previous = history.get(identifier)
                if previous is not None and previous["sequence"] == perception["sequence"]:
                    totals["repeated_frames"] += 1
                    if target["tracking"] != previous["tracking"]:
                        raise AssertionError("Repeated frames changed track confirmation")
                elif previous is not None:
                    if (target["tracking"]["previous_sequence"] != previous["sequence"] or
                            target["tracking"]["observation_count"] != previous["tracking"]["observation_count"] + 1):
                        raise AssertionError("Track confirmation requires distinct original sequences")
                history[identifier] = {"sequence": perception["sequence"], "tracking": target["tracking"]}
                if source != "models" or target["tracking"]["status"] != "confirmed":
                    continue
                totals["model_confirmed_observations"] += 1
                mask = bbox_mask(target["bbox_xyxy"], (segmentation.shape[1], segmentation.shape[0]))
                counts = {}
                for geom_id, name in enumerate(row["geom_names"]):
                    matches = [asset for asset in APARTMENT_OBJECTS if name == asset or name.startswith(asset + "_")]
                    if matches:
                        asset = matches[0]
                        count = int(np.count_nonzero(mask & (segmentation[..., 0] == geom_id)
                                                     & (segmentation[..., 1] == 5)))
                        counts[asset] = counts.get(asset, 0) + count
                ordered = sorted(((count, asset) for asset, count in counts.items() if count >= 8), reverse=True)
                if not ordered or len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
                    continue
                native_instance = ordered[0][1]
                if identifier in model_instances and model_instances[identifier] != native_instance:
                    raise AssertionError("A model track changed its dominant native instance")
                model_instances[identifier] = native_instance
                model_sequences.add((identifier, perception["sequence"]))
                totals["native_pixel_identities"] += 1
            if perception["targets"]:
                altered = deepcopy(perception)
                altered["sequence"] = True
                try:
                    tracker.update(altered, source)
                except ValueError:
                    totals["invalid_measurements_rejected"] += 1
                else:
                    raise AssertionError("Tracking accepted a boolean physical sequence")
                measured = [index for index, target in enumerate(perception["targets"])
                            if target["distance_status"] == "valid"]
                if measured:
                    altered = deepcopy(perception)
                    altered["targets"][measured[0]]["surface_position_world_m"][0] += 0.01
                    try:
                        tracker.update(altered, source)
                    except ValueError:
                        totals["invalid_measurements_rejected"] += 1
                    else:
                        raise AssertionError("Tracking accepted changed geometry within one sequence")
            totals["captures"] += 1
        queries = [item["result"] for item in tools if item["operation"] == "read_track"]
        if not any(query["status"] == "current" for query in queries) or not any(query["status"] == "stale" for query in queries):
            raise AssertionError("Native track queries require current and stale physical records")
        for query in queries:
            if (query["current_target"] is not None) != (query["status"] == "current"):
                raise AssertionError("Historical track status supplied a current target")
            if query["current_target"] is not None and query["last_observation"]["sequence"] != query["sequence"]:
                raise AssertionError("Current target requires the current physical sequence")
        if source == "simulator_ground_truth":
            motion = result["object_motion"]
            if motion["translation_m"] <= 0.02 or not np.isclose(np.linalg.norm(
                    np.asarray(motion["final_position_m"]) - motion["start_position_m"]), motion["translation_m"]):
                raise AssertionError("Native dynamic tracking requires actual physical object movement")
        for frame in json.loads((directory / "frames.json").read_text()):
            data = (directory / frame["file"]).read_bytes()
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                if np.asarray(image).std() == 0:
                    raise AssertionError("Tracking observer frame is uniform")
            totals["observer_frames"] += 1
        for path in directory.iterdir():
            if path.is_file():
                digests[str(path.relative_to(capture))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(model_sequences) < 2 or not model_instances:
        raise AssertionError("Tracking requires repeated original model observations with native instance pixels")
    return {"passed": True, **totals, "model_native_instances": model_instances, "input_sha256": digests,
            "policy_audits": policy_audits,
            "scope": "Native tracking queries, actual CPU YOLO association and physically moving native ball",
            "autonomous_following_acceptance_performed": False, "gpu_acceptance_performed": False}


def main():
    parser = argparse.ArgumentParser(description="复核原生目标跟踪、实际模型图像和实例像素")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="Recompute every admitted action using the original ONNX")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.capture.resolve(strict=True), args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "captures", "native_pixel_identities",
                                                "model_confirmed_observations", "invalid_measurements_rejected")}))


if __name__ == "__main__":
    main()
