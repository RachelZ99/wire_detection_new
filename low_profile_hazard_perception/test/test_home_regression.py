import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from low_profile_hazard_perception.detection_profile import (
    DetectionProfile,
    default_detection_profile_path,
)
from low_profile_hazard_perception.home_regression import (
    GateDecision,
    default_home_regression_manifest_path,
    evaluate_home_regression,
    main,
    render_home_regression_decision,
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": "home-rule-v1",
        "validation_phase": "home_feasibility",
        "profile_id": "dcw2-home-640x360-v1",
        "profile_fingerprint": "profile-fingerprint",
        "rule_version": "training-free-thin-line-v1",
        "thresholds": {
            "minimum_event_recall": {
                "cable": 1.0,
                "obvious_protrusion": 1.0,
            },
            "maximum_persistent_false_events": 0,
            "maximum_confirmation_latency_ms": 350.0,
            "maximum_health_failures": 0,
            "maximum_processing_p95_ms": 80.0,
            "maximum_depth_geometry_average_cpu_cores": 1.0,
            "maximum_memory_growth_bytes": 33554432,
            "maximum_pending_work": 1,
        },
        "required_acceptance_coverage": {
            "hazard_layout": ["straight", "raised"],
            "hazard_kind": ["cable", "obvious_protrusion"],
            "cable_appearance": ["pink_rubber", "white_pvc"],
            "negative_class": ["reflection", "empty_floor"],
            "distance": ["near", "far"],
            "floor": ["reflective_tile"],
            "light": ["indoor_diffuse"],
            "robot_motion": ["straight", "turning"],
            "depth_validity": ["mostly_valid", "cable_invalid"],
        },
        "npu_eligible_failure_classes": [
            "low_contrast_cable",
            "reflection_cable_confusion",
        ],
        "scenes": [
            {
                "scene_id": "tune-pink",
                "scene_group_id": "video-tune-pink",
                "split": "tuning",
                "bag_id": "tune-pink-bag",
                "duration_seconds": 20.0,
                "maximum_speed_mps": 0.1,
                "strata": {
                    "hazard_layout": ["straight"],
                    "hazard_kind": ["cable"],
                    "cable_appearance": ["pink_rubber"],
                    "negative_class": [],
                    "distance": "near",
                    "floor": "reflective_tile",
                    "light": "indoor_diffuse",
                    "robot_motion": "straight",
                    "depth_validity": "mostly_valid",
                },
                "expected_events": [
                    {
                        "event_id": "tune-cable",
                        "hazard_kind": "cable",
                        "cable_appearance": "pink_rubber",
                    }
                ],
                "result_file": "tune-pink.json",
            },
            {
                "scene_id": "accept-straight",
                "scene_group_id": "video-accept-straight",
                "split": "acceptance",
                "bag_id": "accept-straight-bag",
                "duration_seconds": 30.0,
                "maximum_speed_mps": 0.3,
                "strata": {
                    "hazard_layout": ["straight"],
                    "hazard_kind": ["cable"],
                    "cable_appearance": ["pink_rubber"],
                    "negative_class": ["reflection"],
                    "distance": "far",
                    "floor": "reflective_tile",
                    "light": "indoor_diffuse",
                    "robot_motion": "straight",
                    "depth_validity": "mostly_valid",
                },
                "expected_events": [
                    {
                        "event_id": "pink-straight",
                        "hazard_kind": "cable",
                        "cable_appearance": "pink_rubber",
                    }
                ],
                "result_file": "accept-straight.json",
            },
            {
                "scene_id": "accept-turning",
                "scene_group_id": "video-accept-turning",
                "split": "acceptance",
                "bag_id": "accept-turning-bag",
                "duration_seconds": 30.0,
                "maximum_speed_mps": 0.3,
                "strata": {
                    "hazard_layout": ["raised"],
                    "hazard_kind": ["cable", "obvious_protrusion"],
                    "cable_appearance": ["white_pvc"],
                    "negative_class": ["empty_floor"],
                    "distance": "near",
                    "floor": "reflective_tile",
                    "light": "indoor_diffuse",
                    "robot_motion": "turning",
                    "depth_validity": "cable_invalid",
                },
                "expected_events": [
                    {
                        "event_id": "white-raised",
                        "hazard_kind": "cable",
                        "cable_appearance": "white_pvc",
                    },
                    {
                        "event_id": "power-strip",
                        "hazard_kind": "obvious_protrusion",
                    },
                ],
                "result_file": "accept-turning.json",
            },
        ],
    }


def _result(scene_id: str, event_ids: list[str]) -> dict[str, object]:
    scene_evidence = {
        "tune-pink": ("tune-pink-bag", 0.1, 20.0),
        "accept-straight": ("accept-straight-bag", 0.3, 30.0),
        "accept-turning": ("accept-turning-bag", 0.3, 30.0),
    }
    bag_id, observed_speed, evaluated_duration = scene_evidence[scene_id]
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "bag_id": bag_id,
        "bag_sha256": "a" * 64,
        "profile_id": "dcw2-home-640x360-v1",
        "profile_fingerprint": "profile-fingerprint",
        "rule_version": "training-free-thin-line-v1",
        "deterministic": True,
        "repeat_count": 2,
        "evaluated_duration_seconds": evaluated_duration,
        "observed_maximum_speed_mps": observed_speed,
        "event_outcomes": [
            {
                "event_id": event_id,
                "detected": True,
                "confirmed_detection_distance_m": 0.8,
                "confirmation_latency_ms": 100.0,
                "failure_class": None,
            }
            for event_id in event_ids
        ],
        "persistent_false_events": [],
        "health_failures": [],
        "resource_use": {
            "processing_p95_ms": 70.0,
            "depth_geometry_average_cpu_cores": 0.8,
            "memory_growth_bytes": 1024,
            "maximum_pending_work": 1,
        },
    }


def _passing_results() -> dict[str, dict[str, object]]:
    return {
        "tune-pink": _result("tune-pink", ["tune-cable"]),
        "accept-straight": _result(
            "accept-straight", ["pink-straight"]
        ),
        "accept-turning": _result(
            "accept-turning", ["white-raised", "power-strip"]
        ),
    }


class HomeRegressionTests(unittest.TestCase):
    def test_complete_held_out_suite_passes_rule_path_and_reports_strata(
        self,
    ) -> None:
        report = evaluate_home_regression(_manifest(), _passing_results())

        self.assertEqual(report["decision"], GateDecision.RULE_PATH_PASSES.value)
        self.assertFalse(report["execute_ticket_8"])
        self.assertEqual(report["acceptance_metrics"]["event_recall"], 1.0)
        self.assertEqual(
            report["acceptance_metrics"]["persistent_false_event_count"], 0
        )
        self.assertEqual(
            report["stratified"]["robot_motion"]["turning"][
                "expected_event_count"
            ],
            2,
        )
        self.assertEqual(
            report["stratified"]["cable_appearance"]["white_pvc"][
                "expected_event_count"
            ],
            1,
        )
        self.assertEqual(report["validation_phase"], "home_feasibility")
        self.assertFalse(report["factory_validation"])

    def test_specific_rgb_rule_failure_selects_npu(self) -> None:
        results = _passing_results()
        missed = results["accept-turning"]["event_outcomes"][0]
        missed.update(
            detected=False,
            confirmed_detection_distance_m=None,
            confirmation_latency_ms=None,
            failure_class="low_contrast_cable",
        )

        report = evaluate_home_regression(_manifest(), results)

        self.assertEqual(report["decision"], GateDecision.NPU_REQUIRED.value)
        self.assertTrue(report["execute_ticket_8"])
        self.assertEqual(
            report["npu_failure_classes"], ["low_contrast_cable"]
        )

    def test_specific_persistent_false_event_selects_npu(self) -> None:
        results = _passing_results()
        results["accept-straight"]["persistent_false_events"] = [
            {
                "event_id": "false-reflection",
                "failure_class": "reflection_cable_confusion",
                "duration_seconds": 2.0,
            }
        ]

        report = evaluate_home_regression(_manifest(), results)

        self.assertEqual(report["decision"], GateDecision.NPU_REQUIRED.value)
        self.assertEqual(
            report["npu_failure_classes"],
            ["reflection_cable_confusion"],
        )

    def test_missing_evidence_blocks_without_claiming_npu_or_rule_pass(
        self,
    ) -> None:
        results = _passing_results()
        del results["accept-turning"]

        report = evaluate_home_regression(_manifest(), results)

        self.assertEqual(report["decision"], GateDecision.EVIDENCE_INCOMPLETE.value)
        self.assertFalse(report["execute_ticket_8"])
        self.assertIn("missing_result:accept-turning", report["blocking_reasons"])

    def test_planned_full_speed_does_not_replace_observed_full_speed(self) -> None:
        results = _passing_results()
        results["accept-turning"]["observed_maximum_speed_mps"] = 0.14

        report = evaluate_home_regression(_manifest(), results)

        self.assertEqual(report["decision"], GateDecision.EVIDENCE_INCOMPLETE.value)
        self.assertIn(
            "full_speed_recording_missing:turning", report["blocking_reasons"]
        )

    def test_obvious_protrusion_miss_does_not_misroute_to_npu(self) -> None:
        results = _passing_results()
        missed = results["accept-turning"]["event_outcomes"][1]
        missed.update(
            detected=False,
            confirmed_detection_distance_m=None,
            confirmation_latency_ms=None,
            failure_class="geometry_low_support",
        )

        report = evaluate_home_regression(_manifest(), results)

        self.assertEqual(report["decision"], GateDecision.NON_NPU_FAILURE.value)
        self.assertFalse(report["execute_ticket_8"])
        self.assertIn(
            "obvious_protrusion_recall_below_gate",
            report["blocking_reasons"],
        )

    def test_scene_group_cannot_leak_between_tuning_and_acceptance(self) -> None:
        manifest = copy.deepcopy(_manifest())
        manifest["scenes"][1]["scene_group_id"] = "video-tune-pink"

        with self.assertRaisesRegex(ValueError, "scene_group_id"):
            evaluate_home_regression(manifest, _passing_results())

    def test_decision_record_is_explicitly_home_only(self) -> None:
        markdown = render_home_regression_decision(
            evaluate_home_regression(_manifest(), _passing_results())
        )

        self.assertIn("RULE_PATH_PASSES", markdown)
        self.assertIn("Do not execute ticket 8", markdown)
        self.assertIn("Home feasibility evidence", markdown)
        self.assertIn("not factory validation", markdown)
        self.assertIn("Event recall", markdown)
        self.assertIn("Confirmed detection distance", markdown)
        self.assertIn("Confirmation latency", markdown)
        self.assertIn("Resource use", markdown)

    def test_cli_loads_external_scene_results_and_writes_both_reports(
        self,
    ) -> None:
        manifest = _manifest()
        results = _passing_results()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_path = root / "report.json"
            decision_path = root / "decision.md"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for scene in manifest["scenes"]:
                (root / scene["result_file"]).write_text(
                    json.dumps(results[scene["scene_id"]]), encoding="utf-8"
                )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(manifest_path),
                        "--results-directory",
                        str(root),
                        "--output",
                        str(output_path),
                        "--decision-record",
                        str(decision_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "RULE_PATH_PASSES")
            self.assertIn(
                "not factory validation",
                decision_path.read_text(encoding="utf-8"),
            )

    def test_repository_manifest_covers_the_ticket_without_fake_results(
        self,
    ) -> None:
        manifest = json.loads(
            default_home_regression_manifest_path().read_text(encoding="utf-8")
        )

        report = evaluate_home_regression(manifest, {})
        profile = DetectionProfile.load(default_detection_profile_path())

        self.assertEqual(report["coverage"]["missing"], {})
        self.assertEqual(manifest["profile_fingerprint"], profile.fingerprint)
        self.assertEqual(report["decision"], "EVIDENCE_INCOMPLETE")
        self.assertFalse(report["factory_validation"])
        self.assertIn(
            "missing_result:accept-empty-floor", report["blocking_reasons"]
        )


if __name__ == "__main__":
    unittest.main()
