from datetime import datetime
import math


def audit_target_tracking(tool_results: list[dict]) -> dict:
    observations, queries = [], []
    by_track = {}
    for detail in tool_results:
        tool, result = detail["tool"], detail["result"]
        if tool == "microduck.inspect_scene":
            observations.append(result)
        elif tool == "microduck.observe" and result.get("perception") is not None:
            perception = result["perception"]
            if any(perception[key] != result[key] for key in ("episode_id", "sequence")):
                raise AssertionError("Tracked observation differs from its physical frame")
            observations.append(perception)
        if tool != "microduck.read_track":
            continue
        last = result["last_observation"]
        target = last["target"]
        matching = [observed for observed in observations
                    if (observed["episode_id"], observed["sequence"], observed["observed_at"], observed["prompt"]) ==
                       (result["episode_id"], last["sequence"], last["observed_at"], result["prompt"])
                    and target in observed["targets"]]
        if (not matching or target["track_id"] != result["track_id"] or
                type(result["sequence"]) is not int or last["sequence"] > result["sequence"] or
                type(result["max_age_ms"]) is not int or not 100 <= result["max_age_ms"] <= 60000 or
                type(result["age_ms"]) not in (int, float) or not math.isfinite(result["age_ms"]) or result["age_ms"] < 0 or
                datetime.fromisoformat(last["observed_at"]).utcoffset() is None):
            raise AssertionError("Track query lacks its original measured target")
        if result["source"] == "models":
            if not any(observed["models"] == result["models"] for observed in matching):
                raise AssertionError("Track query changed its original model identities")
        elif result["source"] != "simulator_ground_truth" or not any(
                observed["detection_source"] == "simulator_ground_truth" for observed in matching):
            raise AssertionError("Track query changed its perception source")
        if result["status"] == "current":
            if (result["current_target"] != target or target["tracking"]["status"] != "confirmed" or
                    last["sequence"] != result["sequence"] or target["distance_status"] != "valid" or
                    not 0 <= result["age_ms"] <= result["max_age_ms"] <= 60000 or
                    result["source"] == "models" and target["tracking"]["consecutive_observations"] < 2):
                raise AssertionError("Current track query lacks a confirmed current measurement")
            by_track.setdefault(result["track_id"], set()).add(result["sequence"])
        elif (result["status"] not in {"tentative", "missing", "ambiguous", "stale", "expired"}
                or result["current_target"] is not None):
            raise AssertionError("Historical track query supplied a current target")
        queries.append(result)
    if not queries or not any(len(sequences) >= 2 for sequences in by_track.values()):
        raise AssertionError("Tracking task requires the same target queried at two physical sequences")
    return {"queries": len(queries), "current_queries": sum(query["status"] == "current" for query in queries),
            "track_sequences": {identifier: sorted(sequences) for identifier, sequences in by_track.items()},
            "sources": sorted({query["source"] for query in queries})}
