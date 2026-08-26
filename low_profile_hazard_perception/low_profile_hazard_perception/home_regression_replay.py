"""Black-box rosbag runner for the event-level home regression gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .home_regression import (
    _json_object,
    _validate_manifest,
    default_home_regression_manifest_path,
    main as audit_main,
)
from .temporal import EvidenceMask


def normalize_home_scene_replay(
    *,
    suite: Mapping[str, object],
    scene: Mapping[str, object],
    annotation: Mapping[str, object],
    runs: Sequence[Mapping[str, Any]],
    bag_sha256: str,
    failure_runs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Turn repeated operational-output captures into one scene result."""
    if len(runs) < 2:
        raise ValueError("home scene replay requires at least two runs")
    scene_id = str(scene["scene_id"])
    annotation_errors = list(_annotation_validator().iter_errors(annotation))
    if annotation_errors:
        first = min(
            annotation_errors,
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        path = ".".join(str(part) for part in first.absolute_path) or "root"
        raise ValueError(
            f"scene {scene_id} annotation schema invalid at {path}: "
            f"{first.message}"
        )
    if annotation.get("schema_version") != 1:
        raise ValueError(f"scene {scene_id} has unsupported annotation schema")
    if annotation.get("scene_id") != scene_id:
        raise ValueError(f"scene {scene_id} annotation identity mismatch")
    canonical = _canonical_run(runs[0])
    deterministic = all(_canonical_run(run) == canonical for run in runs[1:])
    event_annotations = _annotation_events(scene, annotation)
    baseline = runs[0]
    hazard_clouds = _hazard_groups(baseline["clouds"])
    event_hazards = _unique_hazards(hazard_clouds)
    outcomes = [
        _event_outcome(event, event_annotations[event["event_id"]], event_hazards)
        for event in scene["expected_events"]
    ]
    false_events = _persistent_false_events(
        annotation.get("negative_regions", []), hazard_clouds
    )
    health = baseline["health"]
    message_age_fields = {
        key: float(values["maximum"])
        for key, values in health["timing_ranges_ms"].items()
        if key.endswith("sensor_stamp_age_ms")
        or key.endswith("latest_message_age_ms")
    }
    if not message_age_fields:
        raise ValueError(f"scene {scene_id} replay reported no message age")
    transitions = [dict(item) for item in health["transitions"]]
    health_failures = [
        f"{item['state']}:{item['reasons']}"
        for item in transitions
        if item["state"] == "INVALID"
    ]
    final_state = health["canonical"]["state"]
    if final_state != "HEALTHY":
        health_failures.append(
            f"final_{final_state}:{health['canonical']['stable_values'].get('reasons', '')}"
        )
    runtime = baseline["runtime"]
    runtime_resource = runtime.get("resource_use")
    if isinstance(runtime_resource, Mapping):
        resource_use = {
            "processing_p95_ms": _required_number(
                runtime_resource, "processing_p95_ms", scene_id
            ),
            "depth_geometry_average_cpu_cores": _required_number(
                runtime_resource,
                "depth_geometry_average_cpu_cores",
                scene_id,
            ),
            "memory_growth_bytes": int(
                _required_number(
                    runtime_resource, "memory_growth_bytes", scene_id
                )
            ),
            "maximum_pending_work": int(
                _required_number(
                    runtime_resource, "maximum_pending_work", scene_id
                )
            ),
        }
    else:
        volatile = health["latest_volatile_values"]
        stable = health["canonical"]["stable_values"]
        pending_values = [
            int(float(value))
            for key, value in stable.items()
            if key.endswith("pending_count")
        ]
        resource_use = {
            "processing_p95_ms": _required_number(
                volatile,
                "stage.perception.processing_wall_p95_ms",
                scene_id,
            ),
            "depth_geometry_average_cpu_cores": _required_number(
                volatile,
                "stage.depth_geometry.average_cpu_cores",
                scene_id,
            ),
            "memory_growth_bytes": int(
                _required_number(
                    volatile, "resource.memory_growth_bytes", scene_id
                )
            ),
            "maximum_pending_work": max(pending_values, default=0),
        }
    duration = runtime.get("evaluated_duration_seconds")
    if duration is None:
        raise ValueError(f"scene {scene_id} replay reported no evaluated duration")
    injection_outcomes = _failure_injection_outcomes(
        annotation.get("failure_injections", []), failure_runs or {}
    )
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "bag_id": scene["bag_id"],
        "bag_sha256": bag_sha256,
        "profile_id": suite["profile_id"],
        "profile_fingerprint": suite["profile_fingerprint"],
        "rule_version": suite["rule_version"],
        "deterministic": deterministic,
        "repeat_count": len(runs),
        "evaluated_duration_seconds": float(duration),
        "observed_maximum_speed_mps": float(
            runtime["observed_maximum_speed_mps"]
        ),
        "message_age_ms": {
            "maximum": max(message_age_fields.values()),
            "fields": dict(sorted(message_age_fields.items())),
        },
        "health_transitions": transitions,
        "failure_injection_outcomes": injection_outcomes,
        "event_outcomes": outcomes,
        "persistent_false_events": false_events,
        "health_failures": sorted(set(health_failures)),
        "resource_use": resource_use,
    }


@lru_cache(maxsize=1)
def _annotation_validator() -> Draft202012Validator:
    schema_path = default_home_regression_manifest_path().with_name(
        "home_regression_scene_annotation_schema_v1.json"
    )
    schema = _json_object(
        schema_path.read_bytes(), label="home regression annotation schema"
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _canonical_run(run: Mapping[str, Any]) -> dict[str, object]:
    return {
        "health": run["health"]["canonical"],
        "clouds": run["clouds"],
        "runtime": {
            "evaluated_duration_seconds": run["runtime"][
                "evaluated_duration_seconds"
            ],
            "observed_maximum_speed_mps": run["runtime"][
                "observed_maximum_speed_mps"
            ],
        },
    }


def _hazard_groups(
    clouds: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    groups: list[Mapping[str, object]] = []
    for cloud in clouds:
        if cloud.get("clearing"):
            continue
        raw_groups = cloud.get("hazard_groups")
        if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, str):
            groups.extend(
                group for group in raw_groups if isinstance(group, Mapping)
            )
        else:
            groups.append(cloud)
    return groups


def _unique_hazards(
    groups: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    earliest: dict[object, Mapping[str, object]] = {}
    for index, group in enumerate(groups):
        key = group.get("hazard_track_id", ("untracked", index))
        prior = earliest.get(key)
        if prior is None or int(group.get("source_stamp_max_ns") or 0) < int(
            prior.get("source_stamp_max_ns") or 0
        ):
            earliest[key] = group
    return list(earliest.values())


def _annotation_events(
    scene: Mapping[str, object], annotation: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    raw = annotation["events"]
    assert isinstance(raw, Sequence) and not isinstance(raw, str)
    events = {
        str(item["event_id"]): item
        for item in raw
        if isinstance(item, Mapping)
    }
    expected = {event["event_id"] for event in scene["expected_events"]}
    if set(events) != expected:
        raise ValueError(f"scene {scene['scene_id']} annotation events mismatch")
    event_items = list(events.items())
    for index, (first_id, first) in enumerate(event_items):
        first_x, first_y = (float(value) for value in first["center_odom"])
        for second_id, second in event_items[index + 1 :]:
            second_x, second_y = (
                float(value) for value in second["center_odom"]
            )
            if math.hypot(first_x - second_x, first_y - second_y) <= float(
                first["radius_m"]
            ) + float(second["radius_m"]):
                raise ValueError(
                    f"scene {scene['scene_id']} event regions overlap: "
                    f"{first_id}, {second_id}"
                )
    return events


def _event_outcome(
    expected: Mapping[str, object],
    annotation: Mapping[str, object],
    clouds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    center_x, center_y = (float(value) for value in annotation["center_odom"])
    radius = float(annotation["radius_m"])
    matching = [
        cloud
        for cloud in clouds
        if cloud.get("centroid_x_m") is not None
        and cloud.get("centroid_y_m") is not None
        and math.hypot(
            float(cloud["centroid_x_m"]) - center_x,
            float(cloud["centroid_y_m"]) - center_y,
        )
        <= radius
    ]
    if not matching:
        failure_class = annotation.get("failure_class_if_missed")
        if not isinstance(failure_class, str) or not failure_class:
            failure_class = "unclassified_failure"
        return {
            "event_id": expected["event_id"],
            "detected": False,
            "confirmed_detection_distance_m": None,
            "confirmation_latency_ms": None,
            "failure_class": failure_class,
        }
    cloud = min(
        matching,
        key=lambda item: int(item.get("source_stamp_max_ns") or 0),
    )
    distance = cloud.get("confirmed_detection_distance_m")
    latency = cloud.get("confirmation_latency_ms")
    if distance is None or latency is None:
        raise ValueError(
            f"event {expected['event_id']} cloud lacks distance or latency"
        )
    return {
        "event_id": expected["event_id"],
        "detected": True,
        "confirmed_detection_distance_m": float(distance),
        "confirmation_latency_ms": float(latency),
        "failure_class": None,
    }


def _persistent_false_events(
    raw_regions: object,
    clouds: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, str):
        raise ValueError("negative_regions must be a list")
    events: list[dict[str, object]] = []
    for region in raw_regions:
        if not isinstance(region, Mapping):
            raise ValueError("negative region must be an object")
        bounds = region.get("bounds_odom")
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, str)
            or len(bounds) != 4
        ):
            raise ValueError("negative region bounds_odom must have four values")
        minimum_x, maximum_x, minimum_y, maximum_y = (
            float(value) for value in bounds
        )
        matching = [
            cloud
            for cloud in clouds
            if cloud.get("centroid_x_m") is not None
            and cloud.get("centroid_y_m") is not None
            and minimum_x <= float(cloud["centroid_x_m"]) <= maximum_x
            and minimum_y <= float(cloud["centroid_y_m"]) <= maximum_y
        ]
        matches_by_track: dict[object, list[Mapping[str, object]]] = {}
        for index, cloud in enumerate(matching):
            key = cloud.get("hazard_track_id", ("untracked", index))
            matches_by_track.setdefault(key, []).append(cloud)
        for track_id, track_matches in matches_by_track.items():
            first_observation_ns = min(
                int(cloud.get("source_stamp_max_ns") or 0)
                - int(float(cloud.get("confirmation_latency_ms") or 0.0) * 1_000_000)
                for cloud in track_matches
            )
            last_confirmation_ns = max(
                int(cloud.get("source_stamp_max_ns") or 0)
                for cloud in track_matches
            )
            measured_duration = max(
                0.0,
                (last_confirmation_ns - first_observation_ns) / 1_000_000_000,
            )
            minimum_duration = float(region["minimum_persistence_seconds"])
            if measured_duration + 1e-9 < minimum_duration:
                continue
            has_rgb_evidence = any(
                int(cloud.get("evidence_mask") or 0)
                & int(EvidenceMask.RGB_CABLE)
                for cloud in track_matches
            )
            events.append(
                {
                    "event_id": f"{region['event_id']}:{track_id}",
                    "failure_class": (
                        str(region["failure_class"])
                        if has_rgb_evidence
                        else "geometry_false_positive"
                    ),
                    "duration_seconds": round(measured_duration, 6),
                }
            )
    return events


def _failure_injection_outcomes(
    raw_injections: object,
    runs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    if not isinstance(raw_injections, Sequence) or isinstance(raw_injections, str):
        raise ValueError("failure_injections must be a list")
    outcomes: list[dict[str, object]] = []
    for injection in raw_injections:
        if not isinstance(injection, Mapping):
            raise ValueError("failure injection must be an object")
        name = str(injection.get("injection", ""))
        if not name or name not in runs:
            continue
        run = runs[name]
        state = str(run["health"]["canonical"]["state"])
        new_hazard_observed = any(
            not cloud.get("clearing") for cloud in run["clouds"]
        )
        expected_state = str(injection.get("expected_health_state", "DEGRADED"))
        forbid_hazard = bool(injection.get("forbid_new_confirmed_hazard", True))
        expected_npu_state = injection.get("expected_npu_state")
        actual_npu_state = run["health"]["latest_volatile_values"].get(
            "resource.npu_state"
        )
        failure_observed = _failure_observed(name, run, actual_npu_state)
        passed = failure_observed and state == expected_state and (
            not forbid_hazard or not new_hazard_observed
        )
        if expected_npu_state is not None:
            passed = passed and actual_npu_state == expected_npu_state
        outcomes.append(
            {
                "injection": name,
                "health_state": state,
                "failure_observed": failure_observed,
                "new_confirmed_hazard_observed": new_hazard_observed,
                "passed": passed,
            }
        )
    return outcomes


def _failure_observed(
    injection: str,
    run: Mapping[str, Any],
    actual_npu_state: object,
) -> bool:
    if injection == "npu":
        return actual_npu_state in {
            "disabled_rule_profile",
            "unavailable",
            "failed",
        }
    health = run["health"]
    reason_text = " ".join(
        str(transition.get("reasons", ""))
        for transition in health["transitions"]
    ).lower()
    stable = health["canonical"]["stable_values"]
    reason_text += " " + str(stable.get("reasons", "")).lower()
    if injection == "ground_model":
        return "ground" in reason_text or stable.get("ground.state") in {
            "REJECTED",
            "UNAVAILABLE",
        }
    return injection in reason_text


def _required_number(
    values: Mapping[str, object], key: str, scene_id: str
) -> float:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"scene {scene_id} replay is missing {key}") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"scene {scene_id} replay has invalid {key}")
    return value


def _bag_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"bag contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _ensure_fresh_result_targets(
    results_directory: Path,
    scenes: Sequence[Mapping[str, object]],
) -> None:
    existing = [
        str(scene["result_file"])
        for scene in scenes
        if (results_directory / str(scene["result_file"])).exists()
    ]
    if existing:
        raise ValueError(
            "results directory contains stale manifest result files: "
            + ", ".join(sorted(existing))
        )


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the asynchronous perception graph, replay every home bag "
            "twice, collect operational odom clouds/health, and audit the NPU gate."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=default_home_regression_manifest_path(),
    )
    parser.add_argument("--bags-directory", type=Path, required=True)
    parser.add_argument("--annotations-directory", type=Path, required=True)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-record", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    options = parser.parse_args(args)
    if options.repeat < 2 or options.rate <= 0.0:
        parser.error("repeat must be at least two and rate must be positive")
    try:
        manifest = _json_object(
            options.manifest.read_bytes(), label=str(options.manifest)
        )
        suite = _validate_manifest(manifest)
        options.results_directory.mkdir(parents=True, exist_ok=True)
        _ensure_fresh_result_targets(
            options.results_directory, suite["scenes"]
        )
        import rclpy

        from .geometric_replay import _run_once

        rclpy.init(args=[])
        try:
            for scene in suite["scenes"]:
                bag = options.bags_directory / scene["bag_id"]
                annotation_path = (
                    options.annotations_directory
                    / scene.get("annotation_file", f"{scene['scene_id']}.json")
                )
                if not bag.exists() or not annotation_path.is_file():
                    continue
                annotation = _json_object(
                    annotation_path.read_bytes(), label=str(annotation_path)
                )
                runs = [
                    _run_once(
                        bag,
                        health_topic="/low_profile_hazard_perception/health",
                        cloud_topic=(
                            "/low_profile_hazard_perception/confirmed_hazards"
                        ),
                        rate=options.rate,
                        startup_timeout=options.startup_timeout,
                    )
                    for _ in range(options.repeat)
                ]
                failure_runs: dict[str, Mapping[str, Any]] = {}
                raw_injections = annotation.get("failure_injections", [])
                if not isinstance(raw_injections, Sequence) or isinstance(
                    raw_injections, str
                ):
                    raise ValueError(
                        f"scene {scene['scene_id']} failure_injections must be a list"
                    )
                for injection in raw_injections:
                    if not isinstance(injection, Mapping):
                        raise ValueError("failure injection must be an object")
                    injection_name = str(injection["injection"])
                    if injection_name == "npu":
                        if (
                            injection.get("expected_npu_state")
                            != "disabled_rule_profile"
                        ):
                            raise ValueError(
                                "rule-profile npu injection must expect "
                                "disabled_rule_profile"
                            )
                        failure_runs[injection_name] = runs[0]
                        continue
                    injection_bag = options.bags_directory / str(
                        injection.get("bag_id", scene["bag_id"])
                    )
                    topics = injection.get("playback_topics", [])
                    if not isinstance(topics, Sequence) or isinstance(topics, str):
                        raise ValueError(
                            f"failure injection {injection_name} topics must be a list"
                        )
                    failure_runs[injection_name] = _run_once(
                        injection_bag,
                        health_topic="/low_profile_hazard_perception/health",
                        cloud_topic=(
                            "/low_profile_hazard_perception/confirmed_hazards"
                        ),
                        rate=options.rate,
                        startup_timeout=options.startup_timeout,
                        playback_topics=tuple(str(topic) for topic in topics),
                    )
                result = normalize_home_scene_replay(
                    suite=suite,
                    scene=scene,
                    annotation=annotation,
                    runs=runs,
                    bag_sha256=_bag_sha256(bag),
                    failure_runs=failure_runs,
                )
                (options.results_directory / scene["result_file"]).write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        finally:
            rclpy.shutdown()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return audit_main(
        [
            str(options.manifest),
            "--results-directory",
            str(options.results_directory),
            "--output",
            str(options.output),
            "--decision-record",
            str(options.decision_record),
        ]
    )
