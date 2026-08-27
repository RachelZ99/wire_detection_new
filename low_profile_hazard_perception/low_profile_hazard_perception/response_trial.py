"""Audit an offline or isolated-domain whole-robot response trial log."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ResponseTrialRecorder:
    """Collect versioned trial events without assuming robot topic names."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []

    def record(
        self,
        event: str,
        *,
        stamp_ns: int | None = None,
        **values: object,
    ) -> None:
        item = {
            "event": event,
            "stamp_ns": time.time_ns() if stamp_ns is None else stamp_ns,
            **values,
        }
        _event_stamp(item)
        self._events.append(item)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "events": sorted(self._events, key=_event_stamp),
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.document(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def evaluate_response_trial(
    events: Sequence[Mapping[str, object]],
    *,
    maximum_profile_speed_mps: float = 0.3,
    stopped_speed_mps: float = 0.02,
    braking_speed_drop_mps: float = 0.01,
    odom_pair_tolerance_ns: int = 100_000_000,
) -> dict[str, Any]:
    """Derive latency, braking, and stopping evidence from recorded streams."""
    ordered = sorted((dict(event) for event in events), key=_event_stamp)
    by_kind: dict[str, list[dict[str, object]]] = {}
    for event in ordered:
        kind = str(event.get("event", ""))
        if not kind:
            raise ValueError("trial event is missing event")
        by_kind.setdefault(kind, []).append(event)

    required = (
        "trial_started",
        "confirmed_hazard",
        "unified_response_received",
        "response_started",
        "command_sample",
        "odom_sample",
        "trial_finished",
    )
    reasons = [f"missing:{kind}" for kind in required if not by_kind.get(kind)]
    started = _first(by_kind, "trial_started")
    confirmed = _first(by_kind, "confirmed_hazard")
    response_received = _first(by_kind, "unified_response_received")
    response_started = _first(by_kind, "response_started")
    finished = _first(by_kind, "trial_finished")

    planned_speed = _optional_number(started, "planned_speed_mps")
    observed_speed = max(
        (
            abs(float(event.get("linear_mps", 0.0)))
            for event in ordered
            if event.get("event") in ("command_sample", "odom_sample")
        ),
        default=None,
    )
    if planned_speed is not None and planned_speed > maximum_profile_speed_mps + 1e-9:
        reasons.append("profile_speed_exceeded")
    if observed_speed is not None and observed_speed > maximum_profile_speed_mps + 1e-9:
        reasons.append("observed_speed_exceeded")

    observation_stamp_ns = _optional_integer(confirmed, "observation_stamp_ns")
    confirmed_stamp_ns = _optional_integer(confirmed, "confirmed_stamp_ns")
    response_received_ns = _optional_stamp(response_received)
    response_started_ns = _optional_stamp(response_started)
    track_id = _optional_integer(confirmed, "hazard_track_id")
    for event, label in (
        (response_received, "unified_response_received"),
        (response_started, "response_started"),
    ):
        other_track = _optional_integer(event, "hazard_track_id")
        if track_id is not None and other_track is not None and other_track != track_id:
            reasons.append(f"hazard_track_mismatch:{label}")

    camera_to_response_ms = _elapsed_ms(observation_stamp_ns, response_received_ns)
    camera_to_start_ms = _elapsed_ms(observation_stamp_ns, response_started_ns)
    confirmation_latency_ms = _elapsed_ms(observation_stamp_ns, confirmed_stamp_ns)
    timeline = (
        observation_stamp_ns,
        confirmed_stamp_ns,
        response_received_ns,
        response_started_ns,
    )
    present_timeline = [stamp for stamp in timeline if stamp is not None]
    if present_timeline != sorted(present_timeline):
        reasons.append("timestamp_order_invalid")

    smoothed = [
        event
        for event in by_kind.get("command_sample", [])
        if event.get("stream") == "smoothed_command"
    ]
    initial_speed = None
    if response_started_ns is not None:
        prior = [
            abs(float(event["linear_mps"]))
            for event in smoothed
            if _event_stamp(event) <= response_started_ns
        ]
        if prior:
            initial_speed = max(prior)
    braking = None
    if response_started_ns is not None and initial_speed is not None:
        braking = next(
            (
                event
                for event in smoothed
                if _event_stamp(event) >= response_started_ns
                and abs(float(event["linear_mps"]))
                <= initial_speed - braking_speed_drop_mps
            ),
            None,
        )
    if by_kind.get("command_sample") and braking is None:
        reasons.append("missing:braking_start")
    braking_stamp_ns = _optional_stamp(braking)

    fused = [
        event
        for event in by_kind.get("odom_sample", [])
        if event.get("stream") == "fused_odom"
    ]
    wheel = [
        event
        for event in by_kind.get("odom_sample", [])
        if event.get("stream") == "wheel_odom"
    ]
    braking_pose = _nearest_at_or_before(fused, braking_stamp_ns)
    stopped_pose = None
    if braking_stamp_ns is not None:
        for candidate in fused:
            stamp_ns = _event_stamp(candidate)
            if stamp_ns < braking_stamp_ns:
                continue
            wheel_sample = _nearest(wheel, stamp_ns, odom_pair_tolerance_ns)
            if (
                abs(float(candidate.get("linear_mps", math.inf))) <= stopped_speed_mps
                and abs(float(candidate.get("angular_rps", math.inf)))
                <= stopped_speed_mps
                and wheel_sample is not None
                and abs(float(wheel_sample.get("linear_mps", math.inf)))
                <= stopped_speed_mps
                and abs(float(wheel_sample.get("angular_rps", math.inf)))
                <= stopped_speed_mps
            ):
                stopped_pose = candidate
                break
    if by_kind.get("odom_sample") and stopped_pose is None:
        reasons.append("missing:stopped_pose")
    stopping_distance = _pose_distance(braking_pose, stopped_pose)

    incomplete = any(reason.startswith("missing:") for reason in reasons)
    rejected = any(
        reason.startswith("profile_")
        or reason.startswith("observed_speed")
        or reason.startswith("hazard_track_mismatch")
        or reason.startswith("timestamp_order")
        for reason in reasons
    )
    evidence_state = (
        "INCOMPLETE" if incomplete else "REJECTED" if rejected else "MEASURED"
    )
    health_transitions = [
        {
            "stamp_ns": _event_stamp(event),
            "state": event.get("state"),
            "reasons": event.get("reasons", []),
        }
        for event in by_kind.get("health_transition", [])
    ]
    return {
        "schema_version": 1,
        "evidence_state": evidence_state,
        "trial_id": started.get("trial_id") if started else None,
        "profile_id": started.get("profile_id") if started else None,
        "planned_speed_mps": planned_speed,
        "observed_maximum_speed_mps": observed_speed,
        "motion": started.get("motion") if started else None,
        "approach_angle_degrees": (
            started.get("approach_angle_degrees") if started else None
        ),
        "hazard_kind": started.get("hazard_kind") if started else None,
        "hazard_track_id": track_id,
        "observation_stamp_ns": observation_stamp_ns,
        "confirmed_stamp_ns": confirmed_stamp_ns,
        "unified_response_received_stamp_ns": response_received_ns,
        "response_started_stamp_ns": response_started_ns,
        "braking_start_stamp_ns": braking_stamp_ns,
        "stopped_stamp_ns": _optional_stamp(stopped_pose),
        "camera_to_response_ms": camera_to_response_ms,
        "camera_to_response_start_ms": camera_to_start_ms,
        "confirmation_latency_ms": confirmation_latency_ms,
        "detection_distance_m": _optional_number(confirmed, "detection_distance_m"),
        "braking_start_position": _position(braking_pose),
        "stopping_position": _position(stopped_pose),
        "stopping_distance_m": stopping_distance,
        "command_sample_count": len(by_kind.get("command_sample", [])),
        "wheel_odom_sample_count": len(wheel),
        "fused_odom_sample_count": len(fused),
        "health_transitions": health_transitions,
        "outcome": finished.get("outcome") if finished else None,
        "rejection_reasons": sorted(set(reasons)),
    }


def _event_stamp(event: Mapping[str, object]) -> int:
    try:
        stamp = int(event["stamp_ns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("trial event stamp_ns must be an integer") from error
    if stamp < 0:
        raise ValueError("trial event stamp_ns cannot be negative")
    return stamp


def _first(
    by_kind: Mapping[str, list[dict[str, object]]], kind: str
) -> dict[str, object] | None:
    values = by_kind.get(kind, [])
    return values[0] if values else None


def _optional_number(event: Mapping[str, object] | None, key: str) -> float | None:
    if event is None or event.get(key) is None:
        return None
    try:
        value = float(event[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"trial event {key} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"trial event {key} must be finite")
    return value


def _optional_integer(event: Mapping[str, object] | None, key: str) -> int | None:
    value = _optional_number(event, key)
    return int(value) if value is not None else None


def _optional_stamp(event: Mapping[str, object] | None) -> int | None:
    return _event_stamp(event) if event is not None else None


def _elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return round((end_ns - start_ns) / 1_000_000, 6)


def _nearest_at_or_before(
    events: Sequence[Mapping[str, object]], stamp_ns: int | None
) -> Mapping[str, object] | None:
    if stamp_ns is None:
        return None
    candidates = [event for event in events if _event_stamp(event) <= stamp_ns]
    return max(candidates, key=_event_stamp) if candidates else None


def _nearest(
    events: Sequence[Mapping[str, object]], stamp_ns: int, tolerance_ns: int
) -> Mapping[str, object] | None:
    if not events:
        return None
    candidate = min(events, key=lambda event: abs(_event_stamp(event) - stamp_ns))
    return (
        candidate if abs(_event_stamp(candidate) - stamp_ns) <= tolerance_ns else None
    )


def _position(event: Mapping[str, object] | None) -> dict[str, float] | None:
    if event is None:
        return None
    return {"x_m": float(event["x_m"]), "y_m": float(event["y_m"])}


def _pose_distance(
    start: Mapping[str, object] | None,
    stop: Mapping[str, object] | None,
) -> float | None:
    if start is None or stop is None:
        return None
    return round(
        math.hypot(
            float(stop["x_m"]) - float(start["x_m"]),
            float(stop["y_m"]) - float(start["y_m"]),
        ),
        6,
    )


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit an isolated or physical obstacle-response trial log."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(args)
    try:
        raw = options.input.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(
            document.get("events"), list
        ):
            raise ValueError("trial input must be an object with an events list")
        report = evaluate_response_trial(document["events"])
        report["input_sha256"] = hashlib.sha256(raw).hexdigest()
        options.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["evidence_state"] == "MEASURED" else 1
