from datetime import datetime
import math


def audit_target_approach(tool_results: list[dict], *, required_source: str | None = None) -> dict:
    if required_source not in (None, "models", "simulator_ground_truth"):
        raise ValueError("Target audit requires a supported perception source")
    prepared = {}
    observations = []
    completed = set()
    maximum_error = 0.0
    sources, models = set(), {}
    for detail in tool_results:
        tool, result = detail["tool"], detail["result"]
        if tool == "microduck.inspect_scene":
            observations.append(result)
        elif tool == "microduck.observe" and result.get("perception") is not None:
            perception = result["perception"]
            if (perception["episode_id"], perception["sequence"]) != (result["episode_id"], result["sequence"]):
                raise AssertionError("Combined target observation differs from its physical frame")
            observations.append(perception)
        if tool == "microduck.approach_object":
            identity = result["request_id"]
            if identity in prepared or not result["prepared"]:
                raise AssertionError("Target preparation requires a unique admitted request")
            original = result["target_observation"]
            target = original["target"]
            matching = [observation for observation in observations if
                all(observation[key] == original[key] for key in
                    ("episode_id", "sequence", "observed_at", "prompt")) and
                target in observation["targets"]]
            if not matching:
                raise AssertionError("Target preparation lacks its original recorded perception")
            source = original["source"]
            if required_source is not None and source != required_source:
                raise AssertionError("Target preparation differs from the required perception source")
            if source == "models":
                identities = original.get("models", {})
                if (not identities or not any(observation.get("models") == identities for observation in matching) or
                        any(not isinstance(name, str) or not name or not isinstance(digest, str) or
                            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
                            for name, digest in identities.items()) or
                        target["detection_source"] not in {"yolo26", "sam3.1"} or
                        not any(observation["detection_source"] == target["detection_source"] for observation in matching)):
                    raise AssertionError("Model target preparation lacks its original model identities")
                for name, digest in identities.items():
                    if name in models and models[name] != digest:
                        raise AssertionError("Model identity changed during target approaches")
                    models[name] = digest
            elif source != "simulator_ground_truth" or not any(
                    observation["detection_source"] == "simulator_ground_truth" for observation in matching):
                raise AssertionError("Target preparation changed its original perception source")
            sources.add(source)
            if (target["target_id"] != result["arguments"]["target_id"] or
                    target["distance_status"] != "valid" or target["valid_depth_pixels"] < 8 or
                    result["standoff_m"] != result["arguments"]["standoff_m"] or
                    result["distance_tolerance_m"] != 0.05):
                raise AssertionError("Target preparation differs from its requested RGBD target")
            point, start = target["surface_position_world_m"], result["start_position_m"]
            if (len(point) != 3 or not all(math.isfinite(value) for value in point) or
                    not math.isclose(math.dist(point[:2], start[:2]), target["distance_xy_m"], abs_tol=1e-4) or
                    not 0 <= original["age_ms_at_admission"] <= original["max_age_ms"] <= 60000 or
                    datetime.fromisoformat(original["observed_at"]).utcoffset() is None):
                raise AssertionError("Target preparation lacks current measured world geometry")
            prepared[identity] = result
        if tool not in {"microduck.task_progress", "microduck.observe", "microduck.wait_for_motion"}:
            continue
        motion = result.get("metric_motion")
        if motion is None or motion["operation"] != "approach_object":
            continue
        admission = prepared[motion["request_id"]]
        point = admission["target_observation"]["target"]["surface_position_world_m"]
        measured = math.dist(point[:2], result["body_position_m"][:2])
        error = abs(measured - admission["standoff_m"])
        if (motion["target_observation"] != admission["target_observation"] or
                motion["surface_position_world_m"] != point or
                motion["start_position_m"] != admission["start_position_m"] or
                motion["body_position_m"] != result["body_position_m"] or
                motion["sequence"] != result["sequence"] or
                result["episode_id"] != admission["target_observation"]["episode_id"] or
                not math.isclose(motion["measured"], measured, rel_tol=0, abs_tol=1e-6) or
                not math.isclose(motion["error"], error, rel_tol=0, abs_tol=1e-6) or
                motion["requested"] != admission["standoff_m"] or
                motion["unit"] != "m" or motion["tolerance"] != 0.05):
            raise AssertionError("Target motion differs from its recorded geometry and request")
        if motion["completed"]:
            if (motion["phase"] != "complete" or error > 0.05 or result["fallen"] or
                    motion["stopped_samples"] != result["stopped_samples"] or result["stopped_samples"] < 5 or
                    result["execution"]["state"] not in {"paused", "ended"} or
                    not result["execution"]["device_confirmed"]):
                raise AssertionError("Target completion lacks a measured upright stopped boundary")
            completed.add(motion["request_id"])
            maximum_error = max(maximum_error, error)
    if not completed or completed != set(prepared):
        raise AssertionError("Every admitted target request requires measured completion")
    return {"completed_requests": len(completed), "maximum_error_m": maximum_error,
            "distance_tolerance_m": 0.05, "required_stopped_samples": 5,
            "perception_sources": sorted(sources), "models": models}
