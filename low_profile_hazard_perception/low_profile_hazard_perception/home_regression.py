"""Event-level home-feasibility regression and evidence-driven NPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class GateDecision(str, Enum):
    """Terminal and non-terminal outcomes of the home feasibility gate."""

    RULE_PATH_PASSES = "RULE_PATH_PASSES"
    NPU_REQUIRED = "NPU_REQUIRED"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    NON_NPU_FAILURE = "NON_NPU_FAILURE"


_STRATIFICATION_DIMENSIONS = (
    "distance",
    "cable_appearance",
    "floor",
    "light",
    "robot_motion",
    "depth_validity",
)


def evaluate_home_regression(
    manifest: Mapping[str, object],
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Evaluate held-out scene events without reaching into detector internals."""
    suite = _validate_manifest(manifest)
    scenes = suite["scenes"]
    acceptance_scenes = [
        scene for scene in scenes if scene["split"] == "acceptance"
    ]
    blocking_reasons = _coverage_reasons(suite, acceptance_scenes)
    accepted_results: dict[str, Mapping[str, object]] = {}
    all_results: dict[str, Mapping[str, object]] = {}
    for scene in scenes:
        scene_id = scene["scene_id"]
        result = results.get(scene_id)
        if result is None:
            blocking_reasons.append(f"missing_result:{scene_id}")
            continue
        reasons = _result_contract_reasons(suite, scene, result)
        blocking_reasons.extend(reasons)
        if reasons:
            continue
        all_results[scene_id] = result
        if scene["split"] == "acceptance":
            accepted_results[scene_id] = result

    blocking_reasons.extend(
        _full_speed_result_reasons(suite, acceptance_scenes, accepted_results)
    )

    acceptance_metrics = _aggregate_metrics(
        acceptance_scenes, accepted_results
    )
    tuning_scenes = [scene for scene in scenes if scene["split"] == "tuning"]
    tuning_results = {
        scene["scene_id"]: all_results[scene["scene_id"]]
        for scene in tuning_scenes
        if scene["scene_id"] in all_results
    }
    thresholds = suite["thresholds"]
    npu_eligible = set(suite["npu_eligible_failure_classes"])
    npu_failure_classes: set[str] = set()
    non_npu_reasons: list[str] = []

    if not blocking_reasons:
        recall_by_kind = acceptance_metrics["event_recall_by_hazard_kind"]
        minimum_recall = thresholds["minimum_event_recall"]
        for hazard_kind, minimum in minimum_recall.items():
            recall = recall_by_kind.get(hazard_kind)
            if recall is None or recall >= minimum:
                continue
            if hazard_kind == "cable":
                missed_classes = _missed_failure_classes(
                    acceptance_scenes,
                    accepted_results,
                    hazard_kind="cable",
                )
                _route_failure_classes(
                    missed_classes,
                    npu_eligible=npu_eligible,
                    npu_failure_classes=npu_failure_classes,
                    non_npu_reasons=non_npu_reasons,
                    unclassified_reason="unclassified_cable_failure",
                )
            else:
                non_npu_reasons.append(
                    f"{hazard_kind}_recall_below_gate"
                )

        false_event_count = acceptance_metrics[
            "persistent_false_event_count"
        ]
        if false_event_count > thresholds["maximum_persistent_false_events"]:
            false_classes = _false_failure_classes(
                acceptance_scenes, accepted_results
            )
            _route_failure_classes(
                false_classes,
                npu_eligible=npu_eligible,
                npu_failure_classes=npu_failure_classes,
                non_npu_reasons=non_npu_reasons,
                unclassified_reason="unclassified_persistent_false_event",
            )

        maximum_latency = acceptance_metrics["confirmation_latency_max_ms"]
        if (
            maximum_latency is not None
            and maximum_latency
            > thresholds["maximum_confirmation_latency_ms"]
        ):
            non_npu_reasons.append("confirmation_latency_exceeded")
        if (
            acceptance_metrics["health_failure_count"]
            > thresholds["maximum_health_failures"]
        ):
            non_npu_reasons.append("health_failures_exceeded")
        resource = acceptance_metrics["resource_use"]
        resource_checks = (
            (
                "processing_p95_ms",
                "maximum_processing_p95_ms",
                "processing_p95_exceeded",
            ),
            (
                "depth_geometry_average_cpu_cores",
                "maximum_depth_geometry_average_cpu_cores",
                "depth_cpu_budget_exceeded",
            ),
            (
                "memory_growth_bytes",
                "maximum_memory_growth_bytes",
                "memory_growth_exceeded",
            ),
            (
                "maximum_pending_work",
                "maximum_pending_work",
                "pending_work_unbounded",
            ),
        )
        for metric, limit, reason in resource_checks:
            value = resource[metric]
            if value is not None and value > thresholds[limit]:
                non_npu_reasons.append(reason)

    if blocking_reasons:
        decision = GateDecision.EVIDENCE_INCOMPLETE
    elif non_npu_reasons:
        decision = GateDecision.NON_NPU_FAILURE
        blocking_reasons.extend(non_npu_reasons)
    elif npu_failure_classes:
        decision = GateDecision.NPU_REQUIRED
    else:
        decision = GateDecision.RULE_PATH_PASSES

    return {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "validation_phase": "home_feasibility",
        "factory_validation": False,
        "profile_id": suite["profile_id"],
        "profile_fingerprint": suite["profile_fingerprint"],
        "rule_version": suite["rule_version"],
        "decision": decision.value,
        "execute_ticket_8": decision is GateDecision.NPU_REQUIRED,
        "npu_failure_classes": sorted(npu_failure_classes),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "acceptance_scene_count": len(acceptance_scenes),
        "completed_acceptance_scene_count": len(accepted_results),
        "tuning_metrics": _aggregate_metrics(tuning_scenes, tuning_results),
        "acceptance_metrics": acceptance_metrics,
        "stratified": _stratified_metrics(
            acceptance_scenes, accepted_results
        ),
        "coverage": _coverage_report(suite, acceptance_scenes),
        "evidence_label": (
            "home feasibility evidence; not factory validation"
        ),
    }


def render_home_regression_decision(report: Mapping[str, object]) -> str:
    """Render the gate result as a reviewable decision record."""
    decision = GateDecision(str(report["decision"]))
    metrics = report["acceptance_metrics"]
    resource = metrics["resource_use"]
    if decision is GateDecision.RULE_PATH_PASSES:
        decision_text = (
            "The held-out rule path passes the configured home gate. "
            "Do not execute ticket 8."
        )
    elif decision is GateDecision.NPU_REQUIRED:
        classes = ", ".join(report["npu_failure_classes"])
        decision_text = (
            "Execute ticket 8 for these measured RGB rule failure classes: "
            f"{classes}."
        )
    elif decision is GateDecision.EVIDENCE_INCOMPLETE:
        decision_text = (
            "No NPU decision is recorded because the required held-out "
            "evidence is incomplete. Do not execute ticket 8."
        )
    else:
        decision_text = (
            "The rule path does not pass, but the measured failures are not "
            "addressable by cable segmentation. Do not execute ticket 8."
        )
    blocking = report.get("blocking_reasons") or []
    blocking_lines = (
        "\n".join(f"- `{reason}`" for reason in blocking)
        if blocking
        else "- None"
    )
    strata = report.get("stratified") or {}
    stratum_lines = []
    for dimension in _STRATIFICATION_DIMENSIONS:
        values = sorted((strata.get(dimension) or {}).keys())
        stratum_lines.append(
            f"- {dimension}: {', '.join(values) if values else 'none'}"
        )
    return (
        "# Home regression NPU gate decision\n\n"
        f"- Suite: `{report['suite_id']}`\n"
        f"- Detection profile: `{report['profile_id']}`\n"
        f"- Rule version: `{report['rule_version']}`\n"
        f"- Decision: `{decision.value}`\n\n"
        "## Decision\n\n"
        f"{decision_text}\n\n"
        "## Held-out acceptance results\n\n"
        f"- Event recall: {_format_ratio(metrics['event_recall'])}\n"
        "- Persistent false events: "
        f"{metrics['persistent_false_event_count']}\n"
        "- Confirmed detection distance (minimum): "
        f"{_format_metric(metrics['confirmed_detection_distance_min_m'], 'm')}\n"
        "- Confirmation latency (P95 / maximum): "
        f"{_format_metric(metrics['confirmation_latency_p95_ms'], 'ms')} / "
        f"{_format_metric(metrics['confirmation_latency_max_ms'], 'ms')}\n"
        f"- Health failures: {metrics['health_failure_count']}\n"
        "- Resource use (worst scene): processing P95 "
        f"{_format_metric(resource['processing_p95_ms'], 'ms')}, depth "
        f"{_format_metric(resource['depth_geometry_average_cpu_cores'], 'cores')}, "
        f"memory growth {_format_metric(resource['memory_growth_bytes'], 'bytes')}, "
        f"pending work {_format_metric(resource['maximum_pending_work'], '')}\n\n"
        "## Stratification coverage\n\n"
        + "\n".join(stratum_lines)
        + "\n\n## Blocking reasons\n\n"
        + blocking_lines
        + "\n\n## Scope\n\n"
        "Home feasibility evidence only; this is not factory validation and "
        "does not establish industrial safety performance. Factory data and "
        "factory acceptance remain mandatory.\n"
    )


def main(args: list[str] | None = None) -> int:
    """Audit normalized external scene results and write the NPU decision."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate held-out home scene results at event level and record "
            "whether the rule path passes or measured RGB failures justify NPU work."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=default_home_regression_manifest_path(),
    )
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-record", type=Path, required=True)
    options = parser.parse_args(args)
    try:
        manifest_bytes = options.manifest.read_bytes()
        manifest = _json_object(manifest_bytes, label=str(options.manifest))
        suite = _validate_manifest(manifest)
        results_root = options.results_directory.resolve()
        if not results_root.is_dir():
            raise ValueError(
                f"results directory does not exist: {options.results_directory}"
            )
        results: dict[str, Mapping[str, object]] = {}
        result_fingerprints: dict[str, str] = {}
        for scene in suite["scenes"]:
            result_path = (results_root / scene["result_file"]).resolve()
            if not result_path.is_relative_to(results_root):
                raise ValueError(
                    f"scene {scene['scene_id']} result_file leaves results directory"
                )
            if not result_path.is_file():
                continue
            result_bytes = result_path.read_bytes()
            results[scene["scene_id"]] = _json_object(
                result_bytes, label=str(result_path)
            )
            result_fingerprints[scene["scene_id"]] = hashlib.sha256(
                result_bytes
            ).hexdigest()
        report = evaluate_home_regression(manifest, results)
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        report["result_sha256"] = dict(sorted(result_fingerprints.items()))
        options.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        options.decision_record.write_text(
            render_home_regression_decision(report), encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    decision = GateDecision(report["decision"])
    if decision in (GateDecision.RULE_PATH_PASSES, GateDecision.NPU_REQUIRED):
        return 0
    if decision is GateDecision.NON_NPU_FAILURE:
        return 1
    return 2


def default_home_regression_manifest_path() -> Path:
    """Locate the installed suite plan, with a source-tree test fallback."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = (
            Path(get_package_share_directory("low_profile_hazard_perception"))
            / "config"
            / "home_regression_manifest_v1.json"
        )
        if installed.exists():
            return installed
    except (ImportError, LookupError):
        pass
    source = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "home_regression_manifest_v1.json"
    )
    if not source.exists():
        raise FileNotFoundError("default home regression manifest is not installed")
    return source


def _validate_manifest(manifest: Mapping[str, object]) -> dict[str, Any]:
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("unsupported home regression manifest schema")
    if manifest.get("validation_phase") != "home_feasibility":
        raise ValueError("home regression must remain home_feasibility")
    required_text = (
        "suite_id",
        "profile_id",
        "profile_fingerprint",
        "rule_version",
    )
    for key in required_text:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest {key} must be a non-empty string")
    scenes_raw = manifest.get("scenes")
    if not isinstance(scenes_raw, Sequence) or isinstance(scenes_raw, str):
        raise ValueError("manifest scenes must be a list")
    scenes: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    bag_splits: dict[str, str] = {}
    for raw in scenes_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("each scene must be an object")
        scene = dict(raw)
        scene_id = _required_text(scene, "scene_id", "scene")
        group_id = _required_text(scene, "scene_group_id", scene_id)
        bag_id = _required_text(scene, "bag_id", scene_id)
        _required_text(scene, "result_file", scene_id)
        split = scene.get("split")
        if split not in ("tuning", "acceptance"):
            raise ValueError(f"scene {scene_id} has invalid split")
        if scene_id in scene_ids:
            raise ValueError(f"duplicate scene_id {scene_id}")
        scene_ids.add(scene_id)
        for identifier, prior_split, label in (
            (group_id, group_splits.get(group_id), "scene_group_id"),
            (bag_id, bag_splits.get(bag_id), "bag_id"),
        ):
            if prior_split is not None and prior_split != split:
                raise ValueError(
                    f"{label} {identifier} leaks between tuning and acceptance"
                )
        group_splits[group_id] = split
        bag_splits[bag_id] = split
        duration = float(scene.get("duration_seconds", 0.0))
        speed = float(scene.get("maximum_speed_mps", -1.0))
        if duration <= 0.0 or speed < 0.0:
            raise ValueError(f"scene {scene_id} has invalid duration or speed")
        strata = scene.get("strata")
        if not isinstance(strata, Mapping):
            raise ValueError(f"scene {scene_id} is missing strata")
        for dimension in _STRATIFICATION_DIMENSIONS:
            if dimension not in strata:
                raise ValueError(
                    f"scene {scene_id} is missing stratum {dimension}"
                )
        events = scene.get("expected_events")
        if not isinstance(events, Sequence) or isinstance(events, str):
            raise ValueError(f"scene {scene_id} expected_events must be a list")
        event_ids: set[str] = set()
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError(f"scene {scene_id} has invalid expected event")
            event_id = _required_text(event, "event_id", scene_id)
            if event_id in event_ids:
                raise ValueError(f"scene {scene_id} duplicates event {event_id}")
            event_ids.add(event_id)
            if event.get("hazard_kind") not in (
                "cable",
                "obvious_protrusion",
            ):
                raise ValueError(
                    f"scene {scene_id} event {event_id} has invalid hazard_kind"
                )
            if event.get("hazard_kind") == "cable":
                appearance = _required_text(
                    event, "cable_appearance", f"scene {scene_id} event {event_id}"
                )
                if appearance not in _values(strata.get("cable_appearance")):
                    raise ValueError(
                        f"scene {scene_id} event {event_id} cable_appearance "
                        "is absent from scene strata"
                    )
        scenes.append(scene)
    if not any(scene["split"] == "tuning" for scene in scenes):
        raise ValueError("manifest must include tuning scenes")
    if not any(scene["split"] == "acceptance" for scene in scenes):
        raise ValueError("manifest must include held-out acceptance scenes")

    thresholds = manifest.get("thresholds")
    coverage = manifest.get("required_acceptance_coverage")
    eligible = manifest.get("npu_eligible_failure_classes")
    if not isinstance(thresholds, Mapping):
        raise ValueError("manifest thresholds must be an object")
    if not isinstance(coverage, Mapping):
        raise ValueError("required_acceptance_coverage must be an object")
    if not isinstance(eligible, Sequence) or isinstance(eligible, str):
        raise ValueError("npu_eligible_failure_classes must be a list")
    recall = thresholds.get("minimum_event_recall")
    if not isinstance(recall, Mapping):
        raise ValueError("minimum_event_recall must be an object")
    normalized_thresholds = dict(thresholds)
    normalized_thresholds["minimum_event_recall"] = {
        str(key): float(value) for key, value in recall.items()
    }
    for value in normalized_thresholds["minimum_event_recall"].values():
        if not 0.0 <= value <= 1.0:
            raise ValueError("event recall thresholds must be within [0, 1]")
    numeric_thresholds = (
        "maximum_persistent_false_events",
        "maximum_confirmation_latency_ms",
        "maximum_health_failures",
        "maximum_processing_p95_ms",
        "maximum_depth_geometry_average_cpu_cores",
        "maximum_memory_growth_bytes",
        "maximum_pending_work",
    )
    for key in numeric_thresholds:
        if key not in normalized_thresholds:
            raise ValueError(f"manifest threshold {key} is required")
        normalized_thresholds[key] = float(normalized_thresholds[key])
        if normalized_thresholds[key] < 0.0:
            raise ValueError(f"manifest threshold {key} cannot be negative")
    return {
        **dict(manifest),
        "scenes": scenes,
        "thresholds": normalized_thresholds,
        "required_acceptance_coverage": {
            str(key): tuple(str(item) for item in _sequence(value, str(key)))
            for key, value in coverage.items()
        },
        "npu_eligible_failure_classes": tuple(str(item) for item in eligible),
    }


def _coverage_reasons(
    suite: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]
) -> list[str]:
    report = _coverage_report(suite, scenes)
    return [
        f"acceptance_coverage_missing:{dimension}:{value}"
        for dimension, values in report["missing"].items()
        for value in values
    ]


def _full_speed_result_reasons(
    suite: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, object]],
) -> list[str]:
    required_speed = float(suite.get("required_full_speed_mps", 0.3))
    tolerance = float(suite.get("full_speed_tolerance_mps", 0.0))
    return [
        f"full_speed_recording_missing:{motion}"
        for motion in ("straight", "turning")
        if not any(
            motion in _values(scene["strata"].get("robot_motion"))
            and scene["scene_id"] in results
            and float(
                results[scene["scene_id"]]["observed_maximum_speed_mps"]
            )
            >= required_speed - tolerance - 1e-9
            for scene in scenes
        )
    ]


def _coverage_report(
    suite: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    required = suite["required_acceptance_coverage"]
    observed: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for dimension, required_values in required.items():
        values: set[str] = set()
        for scene in scenes:
            if dimension == "hazard_kind":
                values.update(
                    str(event["hazard_kind"])
                    for event in scene["expected_events"]
                )
            else:
                values.update(_values(scene["strata"].get(dimension)))
        observed[dimension] = sorted(values)
        absent = sorted(set(required_values) - values)
        if absent:
            missing[dimension] = absent
    return {"required": required, "observed": observed, "missing": missing}


def _result_contract_reasons(
    suite: Mapping[str, Any],
    scene: Mapping[str, Any],
    result: Mapping[str, object],
) -> list[str]:
    scene_id = scene["scene_id"]
    reasons: list[str] = []
    if int(result.get("schema_version", 0)) != 1:
        reasons.append(f"unsupported_result_schema:{scene_id}")
    for key in (
        "scene_id",
        "bag_id",
        "profile_id",
        "profile_fingerprint",
        "rule_version",
    ):
        if key == "scene_id":
            expected = scene_id
        elif key == "bag_id":
            expected = scene["bag_id"]
        else:
            expected = suite[key]
        if result.get(key) != expected:
            reasons.append(f"result_identity_mismatch:{scene_id}:{key}")
    bag_sha256 = result.get("bag_sha256")
    if (
        not isinstance(bag_sha256, str)
        or len(bag_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bag_sha256)
    ):
        reasons.append(f"invalid_bag_sha256:{scene_id}")
    if result.get("deterministic") is not True:
        reasons.append(f"non_deterministic_replay:{scene_id}")
    try:
        repeat_count = int(result.get("repeat_count", 0))
    except (TypeError, ValueError):
        repeat_count = 0
    if repeat_count < 2:
        reasons.append(f"insufficient_replay_repeats:{scene_id}")
    for key in ("evaluated_duration_seconds", "observed_maximum_speed_mps"):
        try:
            value = float(result[key])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"missing_scene_metric:{scene_id}:{key}")
            continue
        if value < 0.0 or (key == "evaluated_duration_seconds" and value == 0.0):
            reasons.append(f"invalid_scene_metric:{scene_id}:{key}")
        if (
            key == "observed_maximum_speed_mps"
            and value
            > float(suite.get("maximum_validated_speed_mps", 0.3)) + 1e-9
        ):
            reasons.append(f"observed_speed_exceeds_profile:{scene_id}")
    outcomes = result.get("event_outcomes")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, str):
        return reasons + [f"invalid_event_outcomes:{scene_id}"]
    expected_ids = {event["event_id"] for event in scene["expected_events"]}
    outcome_ids = [
        outcome.get("event_id")
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    ]
    if len(outcome_ids) != len(outcomes) or len(set(outcome_ids)) != len(outcome_ids):
        reasons.append(f"invalid_event_outcomes:{scene_id}")
    if set(outcome_ids) != expected_ids:
        reasons.append(f"event_outcomes_mismatch:{scene_id}")
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        detected = outcome.get("detected")
        if not isinstance(detected, bool):
            reasons.append(f"invalid_event_detection:{scene_id}")
        if detected:
            for key in (
                "confirmed_detection_distance_m",
                "confirmation_latency_ms",
            ):
                try:
                    value = float(outcome[key])
                except (KeyError, TypeError, ValueError):
                    reasons.append(f"missing_event_metric:{scene_id}:{key}")
                    continue
                if value < 0.0:
                    reasons.append(f"invalid_event_metric:{scene_id}:{key}")
    for key in ("persistent_false_events", "health_failures"):
        value = result.get(key)
        if not isinstance(value, Sequence) or isinstance(value, str):
            reasons.append(f"invalid_{key}:{scene_id}")
    resource = result.get("resource_use")
    if not isinstance(resource, Mapping):
        reasons.append(f"missing_resource_use:{scene_id}")
    else:
        for key in (
            "processing_p95_ms",
            "depth_geometry_average_cpu_cores",
            "memory_growth_bytes",
            "maximum_pending_work",
        ):
            try:
                if float(resource[key]) < 0.0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                reasons.append(f"invalid_resource_metric:{scene_id}:{key}")
    return reasons


def _aggregate_metrics(
    scenes: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, object]],
    *,
    included_event_ids: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    expected_by_kind: dict[str, int] = {}
    detected_by_kind: dict[str, int] = {}
    distances: list[float] = []
    latencies: list[float] = []
    false_events: list[Mapping[str, object]] = []
    health_failure_count = 0
    processing: list[float] = []
    depth_cpu: list[float] = []
    memory: list[float] = []
    pending: list[float] = []
    observed_speeds: list[float] = []
    completed_duration = 0.0
    for scene in scenes:
        result = results.get(scene["scene_id"])
        if result is None:
            continue
        completed_duration += float(result["evaluated_duration_seconds"])
        observed_speeds.append(float(result["observed_maximum_speed_mps"]))
        outcomes = {
            outcome["event_id"]: outcome
            for outcome in result["event_outcomes"]
        }
        for expected in scene["expected_events"]:
            if (
                included_event_ids is not None
                and expected["event_id"]
                not in included_event_ids.get(scene["scene_id"], set())
            ):
                continue
            kind = expected["hazard_kind"]
            expected_by_kind[kind] = expected_by_kind.get(kind, 0) + 1
            outcome = outcomes[expected["event_id"]]
            if not outcome["detected"]:
                continue
            detected_by_kind[kind] = detected_by_kind.get(kind, 0) + 1
            distances.append(float(outcome["confirmed_detection_distance_m"]))
            latencies.append(float(outcome["confirmation_latency_ms"]))
        false_events.extend(result["persistent_false_events"])
        health_failure_count += len(result["health_failures"])
        resource = result["resource_use"]
        processing.append(float(resource["processing_p95_ms"]))
        depth_cpu.append(float(resource["depth_geometry_average_cpu_cores"]))
        memory.append(float(resource["memory_growth_bytes"]))
        pending.append(float(resource["maximum_pending_work"]))
    expected_total = sum(expected_by_kind.values())
    detected_total = sum(detected_by_kind.values())
    recall_by_kind = {
        kind: detected_by_kind.get(kind, 0) / count
        for kind, count in sorted(expected_by_kind.items())
        if count
    }
    return {
        "expected_event_count": expected_total,
        "detected_event_count": detected_total,
        "event_recall": (
            detected_total / expected_total if expected_total else None
        ),
        "event_recall_by_hazard_kind": recall_by_kind,
        "persistent_false_event_count": len(false_events),
        "persistent_false_events_per_hour": (
            len(false_events) * 3600.0 / completed_duration
            if completed_duration > 0.0
            else None
        ),
        "confirmed_detection_distance_min_m": min(distances, default=None),
        "confirmed_detection_distance_median_m": _percentile(distances, 0.5),
        "confirmation_latency_max_ms": max(latencies, default=None),
        "confirmation_latency_p95_ms": _percentile(latencies, 0.95),
        "health_failure_count": health_failure_count,
        "observed_maximum_speed_mps": max(observed_speeds, default=None),
        "resource_use": {
            "processing_p95_ms": max(processing, default=None),
            "depth_geometry_average_cpu_cores": max(depth_cpu, default=None),
            "memory_growth_bytes": max(memory, default=None),
            "maximum_pending_work": max(pending, default=None),
        },
    }


def _stratified_metrics(
    scenes: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for dimension in _STRATIFICATION_DIMENSIONS:
        if dimension == "cable_appearance":
            event_ids_by_value: dict[str, dict[str, set[str]]] = {}
            for scene in scenes:
                for event in scene["expected_events"]:
                    appearance = event.get("cable_appearance")
                    if not appearance:
                        continue
                    event_ids_by_value.setdefault(str(appearance), {}).setdefault(
                        scene["scene_id"], set()
                    ).add(event["event_id"])
            report[dimension] = {
                value: _aggregate_metrics(
                    [
                        scene
                        for scene in scenes
                        if scene["scene_id"] in event_ids
                    ],
                    results,
                    included_event_ids=event_ids,
                )
                for value, event_ids in sorted(event_ids_by_value.items())
            }
            continue
        values = sorted(
            {
                value
                for scene in scenes
                for value in _values(scene["strata"].get(dimension))
            }
        )
        report[dimension] = {
            value: _aggregate_metrics(
                [
                    scene
                    for scene in scenes
                    if value in _values(scene["strata"].get(dimension))
                ],
                results,
            )
            for value in values
        }
    return report


def _missed_failure_classes(
    scenes: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, object]],
    *,
    hazard_kind: str,
) -> set[str]:
    classes: set[str] = set()
    for scene in scenes:
        result = results[scene["scene_id"]]
        outcomes = {
            outcome["event_id"]: outcome
            for outcome in result["event_outcomes"]
        }
        for event in scene["expected_events"]:
            outcome = outcomes[event["event_id"]]
            if event["hazard_kind"] == hazard_kind and not outcome["detected"]:
                failure_class = outcome.get("failure_class")
                if failure_class:
                    classes.add(str(failure_class))
    return classes


def _false_failure_classes(
    scenes: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, object]],
) -> set[str]:
    return {
        str(event["failure_class"])
        for scene in scenes
        for event in results[scene["scene_id"]]["persistent_false_events"]
        if isinstance(event, Mapping) and event.get("failure_class")
    }


def _route_failure_classes(
    classes: set[str],
    *,
    npu_eligible: set[str],
    npu_failure_classes: set[str],
    non_npu_reasons: list[str],
    unclassified_reason: str,
) -> None:
    if not classes:
        non_npu_reasons.append(unclassified_reason)
        return
    npu_failure_classes.update(classes & npu_eligible)
    non_npu_reasons.extend(
        f"failure_not_npu_eligible:{failure_class}"
        for failure_class in sorted(classes - npu_eligible)
    )


def _required_text(
    mapping: Mapping[str, object], key: str, context: str
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} {key} must be a non-empty string")
    return value


def _json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root in {label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{label} must be a list")
    return value


def _values(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence):
        return {str(item) for item in value}
    return set()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def _format_ratio(value: object) -> str:
    if value is None:
        return "not available"
    return f"{float(value) * 100.0:.1f}%"


def _format_metric(value: object, unit: str) -> str:
    if value is None:
        return "not available"
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.3f}{suffix}"
