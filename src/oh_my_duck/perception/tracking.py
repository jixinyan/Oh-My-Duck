from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingParameters:
    control_period_s: float = 0.02
    association_distance_m: float = 0.75
    ambiguity_margin_m: float = 0.10
    maximum_speed_m_s: float = 2.0
    maximum_missed_steps: int = 150
    confirmation_observations: int = 2
    maximum_streams: int = 16
    maximum_tracks_per_stream: int = 64

    def __post_init__(self):
        for name in ("control_period_s", "association_distance_m", "ambiguity_margin_m", "maximum_speed_m_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Tracking {name} must be finite and positive")
        if self.ambiguity_margin_m >= self.association_distance_m:
            raise ValueError("Tracking ambiguity margin must be smaller than the association distance")
        for name in ("maximum_missed_steps", "confirmation_observations", "maximum_streams",
                     "maximum_tracks_per_stream"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Tracking {name} must be a positive integer")


@dataclass
class _Track:
    identifier: str
    first_sequence: int
    sequence: int
    observed_at: str
    target: dict
    observations: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    consecutive_observations: int = 1


@dataclass
class _Stream:
    source: str
    prompt: str
    models: dict
    sequence: int = -1
    fingerprint: str = ""
    targets: list = field(default_factory=list)
    tracks: dict[str, _Track] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Tracking observed_at must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Tracking timestamps require an explicit timezone")
    return parsed


class TargetTracker:
    def __init__(self, parameters: TrackingParameters | None = None):
        self.parameters = TrackingParameters() if parameters is None else parameters
        self._counter = 0
        self.clear()

    def clear(self) -> None:
        self._episode_id = None
        self._streams: dict[tuple, _Stream] = {}

    def _validate(self, result: dict, source: str) -> tuple:
        if source not in ("models", "simulator_ground_truth"):
            raise ValueError("Tracking requires an explicit supported perception source")
        if not isinstance(result["episode_id"], str) or not 1 <= len(result["episode_id"]) <= 256:
            raise ValueError("Tracking requires a nonempty episode ID of at most 256 characters")
        if type(result["sequence"]) is not int or result["sequence"] < 0:
            raise ValueError("Tracking sequence must be a nonnegative integer")
        _timestamp(result["observed_at"])
        if not isinstance(result["prompt"], str) or not 1 <= len(result["prompt"]) <= 120:
            raise ValueError("Tracking prompt must contain 1–120 characters")
        identifiers = set()
        if not isinstance(result["targets"], list) or len(result["targets"]) > self.parameters.maximum_tracks_per_stream:
            raise ValueError("Tracking target count exceeds its configured capacity")
        fields = ("target_id", "label", "distance_status", "valid_depth_pixels", "surface_position_world_m",
                  "distance_m", "distance_xy_m", "distance_interval_m", "bearing_deg", "bbox_xyxy",
                  "detection_source", "mask_source", "distance_source")
        measurements = []
        for target in result["targets"]:
            identifier = target["target_id"]
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("Tracking target IDs must be nonempty and unique within a frame")
            identifiers.add(identifier)
            if not isinstance(target["label"], str) or not target["label"].strip():
                raise ValueError("Tracking requires a nonempty target label")
            count = target["valid_depth_pixels"]
            if type(count) is not int or count < 0:
                raise ValueError("Tracking valid depth count must be a nonnegative integer")
            if target["distance_status"] == "valid":
                position = target["surface_position_world_m"]
                if (count < 8 or len(position) != 3 or any(type(value) not in (int, float)
                        or not math.isfinite(value) for value in position)):
                    raise ValueError("Tracking requires three finite world coordinates and valid depth")
            elif target["distance_status"] != "insufficient_valid_depth" or count >= 8:
                raise ValueError("Tracking target depth status is invalid")
            elif "surface_position_world_m" in target:
                raise ValueError("Unmeasured targets must not supply world coordinates")
            measurements.append({key: target[key] for key in fields if key in target})
        models = result["models"] if source == "models" else {}
        if source == "models" and (not isinstance(models, dict) or not models):
            raise ValueError("Model tracking requires recorded model identities")
        fingerprint = hashlib.sha256(json.dumps(measurements, sort_keys=True,
            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        return (source, result["prompt"], tuple(sorted(models.items()))), fingerprint, models

    def update(self, result: dict, source: str) -> None:
        key, fingerprint, models = self._validate(result, source)
        if self._episode_id != result["episode_id"]:
            self.clear()
            self._episode_id = result["episode_id"]
        stream = self._streams.get(key)
        if stream is None:
            if len(self._streams) >= self.parameters.maximum_streams:
                raise RuntimeError("Tracking stream capacity exceeded")
            stream = _Stream(source, result["prompt"], deepcopy(models))
            self._streams[key] = stream
        sequence = result["sequence"]
        if sequence < stream.sequence:
            raise ValueError("Tracking cannot accept an earlier physical sequence")
        if sequence == stream.sequence:
            if fingerprint != stream.fingerprint:
                raise ValueError("Tracking measurements changed within one physical sequence")
            for target, retained in zip(result["targets"], stream.targets, strict=True):
                target.update(track_id=retained["track_id"], tracking=deepcopy(retained["tracking"]))
            self._annotate_result(result, source)
            return
        tracks = {identifier: deepcopy(track) for identifier, track in stream.tracks.items()
                  if sequence - track.sequence <= self.parameters.maximum_missed_steps}
        targets = deepcopy(result["targets"])
        valid = [index for index, target in enumerate(targets) if target["distance_status"] == "valid"]
        assignments, ambiguous_targets, ambiguous_tracks = {}, {}, set()
        if source == "simulator_ground_truth":
            by_instance = {track.target["target_id"]: identifier for identifier, track in tracks.items()}
            for index in valid:
                identifier = by_instance.get(targets[index]["target_id"])
                if identifier is not None:
                    if targets[index]["label"] != tracks[identifier].target["label"]:
                        raise ValueError("A native instance changed its target label")
                    assignments[index] = (identifier, None)
        elif valid and tracks:
            assignments, ambiguous_targets, ambiguous_tracks = self._associate(targets, valid, tracks, sequence)
        unmatched = [index for index in valid if index not in assignments and index not in ambiguous_targets]
        if len(tracks) + len(unmatched) > self.parameters.maximum_tracks_per_stream:
            raise RuntimeError("Tracking identity capacity exceeded")
        counter = self._counter
        for index, target in enumerate(targets):
            if index not in valid:
                target.update(track_id=None, tracking={"status": "unmeasured", "observation_count": 0})
            elif index in ambiguous_targets:
                target.update(track_id=None, tracking={"status": "ambiguous", "observation_count": 0,
                    "candidate_track_ids": sorted(ambiguous_targets[index])})
            else:
                identifier, error = assignments.get(index, (None, None))
                previous = None if identifier is None else tracks[identifier]
                if previous is None:
                    counter += 1
                    identifier = f"{self._episode_id}:track:{counter}"
                    track = _Track(identifier, sequence, sequence, result["observed_at"], deepcopy(target))
                else:
                    elapsed = (sequence - previous.sequence) * self.parameters.control_period_s
                    velocity = (np.asarray(target["surface_position_world_m"]) -
                                np.asarray(previous.target["surface_position_world_m"])) / elapsed
                    speed = float(np.linalg.norm(velocity))
                    if speed > self.parameters.maximum_speed_m_s:
                        velocity *= self.parameters.maximum_speed_m_s / speed
                    track = _Track(identifier, previous.first_sequence, sequence, result["observed_at"],
                                   deepcopy(target), previous.observations + 1, velocity,
                                   previous.consecutive_observations + 1 if previous.sequence == stream.sequence else 1)
                status = ("confirmed" if source == "simulator_ground_truth" or
                          track.consecutive_observations >= self.parameters.confirmation_observations else "tentative")
                target.update(track_id=identifier, tracking={"status": status,
                    "observation_count": track.observations, "consecutive_observations": track.consecutive_observations,
                    "first_sequence": track.first_sequence,
                    "previous_sequence": None if previous is None else previous.sequence,
                    "last_sequence": sequence, "association_distance_m": error})
                track.target = deepcopy(target)
                tracks[identifier] = track
        stream.sequence, stream.fingerprint = sequence, fingerprint
        stream.targets, stream.tracks, stream.ambiguous = targets, tracks, ambiguous_tracks
        self._counter = counter
        result["targets"] = deepcopy(targets)
        self._annotate_result(result, source)

    def _associate(self, targets: list, valid: list, tracks: dict, sequence: int) -> tuple:
        identifiers = sorted(tracks)
        costs = np.full((len(valid), len(identifiers)), np.inf)
        for column, identifier in enumerate(identifiers):
            track = tracks[identifier]
            elapsed = (sequence - track.sequence) * self.parameters.control_period_s
            predicted = np.asarray(track.target["surface_position_world_m"]) + track.velocity * elapsed
            for row, index in enumerate(valid):
                if targets[index]["label"] != track.target["label"]:
                    continue
                distance = float(np.linalg.norm(np.asarray(targets[index]["surface_position_world_m"]) - predicted))
                if distance <= self.parameters.association_distance_m:
                    costs[row, column] = distance
        ambiguous_rows, ambiguous_columns = set(), set()
        for matrix, result in ((costs, ambiguous_rows), (costs.T, ambiguous_columns)):
            for index, row in enumerate(matrix):
                distances = np.sort(row[np.isfinite(row)])
                if len(distances) > 1 and distances[1] - distances[0] <= self.parameters.ambiguity_margin_m:
                    result.add(index)
        ambiguous = {valid[row]: {identifiers[column] for column in range(len(identifiers))
                     if np.isfinite(costs[row, column])} for row in range(len(valid))
                     if row in ambiguous_rows or any(np.isfinite(costs[row, column]) for column in ambiguous_columns)}
        ambiguous_ids = {identifier for candidates in ambiguous.values() for identifier in candidates}
        while True:
            connected = {valid[row]: {identifiers[column] for column in range(len(identifiers))
                         if np.isfinite(costs[row, column])} for row in range(len(valid))
                         if any(np.isfinite(costs[row, column]) and identifiers[column] in ambiguous_ids
                                for column in range(len(identifiers)))}
            expanded = {identifier for candidates in connected.values() for identifier in candidates}
            if expanded == ambiguous_ids:
                break
            ambiguous_ids.update(expanded)
            ambiguous.update(connected)
        ambiguous.update(connected)
        # 每个 detection 有独立的未匹配列，数量有限的距离矩阵不会强迫关联。
        penalty = self.parameters.association_distance_m + 1
        assignment_costs = np.concatenate((costs, np.full((len(valid), len(valid)), penalty)), axis=1)
        rows, columns = linear_sum_assignment(assignment_costs)
        assignments = {valid[row]: (identifiers[column], float(costs[row, column]))
                       for row, column in zip(rows, columns, strict=True)
                       if column < len(identifiers) and np.isfinite(costs[row, column])
                       and valid[row] not in ambiguous and identifiers[column] not in ambiguous_ids}
        return assignments, ambiguous, ambiguous_ids

    def _annotate_result(self, result: dict, source: str) -> None:
        result["tracking"] = {"method": "native_instance" if source == "simulator_ground_truth"
                              else "rgbd_world_assignment", **vars(self.parameters)}

    def read(self, track_id: str, *, episode_id: str, sequence: int, max_age_ms: int = 60000) -> dict:
        if not isinstance(track_id, str) or not 1 <= len(track_id) <= 512:
            raise ValueError("track_id must contain 1–512 characters")
        if type(max_age_ms) is not int or not 100 <= max_age_ms <= 60000:
            raise ValueError("max_age_ms must be an integer from 100 to 60000")
        if type(sequence) is not int or sequence < 0 or episode_id != self._episode_id:
            raise ValueError("Track query must identify the current physical episode and sequence")
        found = [(stream, stream.tracks[track_id]) for stream in self._streams.values() if track_id in stream.tracks]
        if len(found) != 1:
            raise ValueError("Unknown or expired track_id in the current episode")
        stream, track = found[0]
        age_ms = (datetime.now(timezone.utc) - _timestamp(track.observed_at)).total_seconds() * 1000
        if sequence < stream.sequence or age_ms < 0:
            raise ValueError("Track observation is ahead of the current physical state or clock")
        if sequence - track.sequence > self.parameters.maximum_missed_steps:
            status = "expired"
        elif track_id in stream.ambiguous:
            status = "ambiguous"
        elif stream.sequence > track.sequence:
            status = "missing"
        elif sequence != track.sequence or age_ms > max_age_ms:
            status = "stale"
        elif track.target["tracking"]["status"] == "tentative":
            status = "tentative"
        else:
            status = "current"
        return {"track_id": track_id, "episode_id": episode_id, "sequence": sequence,
                "source": stream.source, "prompt": stream.prompt, "models": deepcopy(stream.models),
                "status": status, "age_ms": age_ms, "max_age_ms": max_age_ms,
                "last_observation": {"sequence": track.sequence, "observed_at": track.observed_at,
                                     "target": deepcopy(track.target)},
                "current_target": deepcopy(track.target) if status == "current" else None}
