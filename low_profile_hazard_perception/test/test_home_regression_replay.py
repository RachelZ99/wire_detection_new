import unittest

from low_profile_hazard_perception.home_regression_replay import (
    _failure_injection_outcomes,
    _persistent_false_events,
    normalize_home_scene_replay,
)


class HomeRegressionReplayTests(unittest.TestCase):
    def test_black_box_runs_become_event_level_scene_evidence(self) -> None:
        scene = {
            "scene_id": "accept-white",
            "bag_id": "accept-white-bag",
            "duration_seconds": 10.0,
            "expected_events": [
                {
                    "event_id": "white-cable",
                    "hazard_kind": "cable",
                    "cable_appearance": "white_pvc",
                }
            ],
        }
        suite = {
            "profile_id": "dcw2-home-640x360-v1",
            "profile_fingerprint": "profile-fingerprint",
            "rule_version": "training-free-thin-line-v1",
        }
        annotation = {
            "schema_version": 1,
            "scene_id": "accept-white",
            "events": [
                {
                    "event_id": "white-cable",
                    "center_odom": [0.8, 0.1],
                    "radius_m": 0.15,
                    "failure_class_if_missed": "low_contrast_cable",
                }
            ],
            "negative_regions": [],
            "failure_injections": [],
        }
        run = {
            "runtime": {
                "evaluated_duration_seconds": 10.0,
                "observed_maximum_speed_mps": 0.3,
            },
            "health": {
                "canonical": {
                    "state": "HEALTHY",
                    "stable_values": {
                        "queue.pending_depth_count": "0",
                        "queue.pending_rgb_count": "0",
                    },
                },
                "transitions": [{"state": "HEALTHY", "reasons": ""}],
                "invalid_observed": False,
                "timing_ranges_ms": {
                    "rgb_image.sensor_stamp_age_ms": {
                        "minimum": 180.0,
                        "maximum": 218.0,
                        "last": 200.0,
                    }
                },
                "latest_volatile_values": {
                    "stage.perception.processing_wall_p95_ms": "70.0",
                    "stage.depth_geometry.average_cpu_cores": "0.8",
                    "resource.memory_growth_bytes": "1024",
                },
            },
            "clouds": [
                {
                    "clearing": False,
                    "hazard_groups": [
                        {
                            "cloud_group_index": 0,
                            "hazard_track_id": 7,
                            "evidence_mask": 1,
                            "centroid_x_m": 0.81,
                            "centroid_y_m": 0.11,
                            "confirmed_detection_distance_m": 0.72,
                            "confirmation_latency_ms": 100.0,
                        }
                    ],
                }
            ],
        }

        result = normalize_home_scene_replay(
            suite=suite,
            scene=scene,
            annotation=annotation,
            runs=[run, run],
            bag_sha256="a" * 64,
        )

        self.assertTrue(result["deterministic"])
        self.assertEqual(result["event_outcomes"][0]["detected"], True)
        self.assertEqual(
            result["event_outcomes"][0]["confirmed_detection_distance_m"],
            0.72,
        )
        self.assertEqual(result["message_age_ms"]["maximum"], 218.0)
        self.assertEqual(result["health_failures"], [])
        self.assertEqual(result["resource_use"]["maximum_pending_work"], 0)

    def test_false_event_uses_measured_span_and_actual_evidence(self) -> None:
        regions = [
            {
                "event_id": "reflection-zone",
                "bounds_odom": [0.0, 1.0, 0.0, 1.0],
                "failure_class": "reflection_cable_confusion",
                "minimum_persistence_seconds": 0.2,
            }
        ]
        clouds = [
            {
                "hazard_track_id": 3,
                "centroid_x_m": 0.5,
                "centroid_y_m": 0.5,
                "source_stamp_max_ns": 1_100_000_000,
                "confirmation_latency_ms": 100.0,
                "evidence_mask": 1,
            },
            {
                "hazard_track_id": 3,
                "centroid_x_m": 0.5,
                "centroid_y_m": 0.5,
                "source_stamp_max_ns": 1_300_000_000,
                "confirmation_latency_ms": 100.0,
                "evidence_mask": 1,
            },
        ]

        events = _persistent_false_events(regions, clouds)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["duration_seconds"], 0.3)
        self.assertEqual(events[0]["failure_class"], "geometry_false_positive")

    def test_one_hazard_track_is_only_counted_once(self) -> None:
        # Reconfirmation republishes the same operational hazard; it is still
        # one event for recall.
        scene = {
            "scene_id": "accept-one",
            "bag_id": "bag",
            "duration_seconds": 1.0,
            "expected_events": [
                {
                    "event_id": "cable",
                    "hazard_kind": "cable",
                    "cable_appearance": "white_pvc",
                }
            ],
        }
        suite = {
            "profile_id": "profile",
            "profile_fingerprint": "fingerprint",
            "rule_version": "rule",
        }
        annotation = {
            "schema_version": 1,
            "scene_id": "accept-one",
            "events": [
                {
                    "event_id": "cable",
                    "center_odom": [0.5, 0.0],
                    "radius_m": 0.1,
                    "failure_class_if_missed": "low_contrast_cable",
                }
            ],
            "negative_regions": [],
            "failure_injections": [],
        }
        health = {
            "canonical": {"state": "HEALTHY", "stable_values": {}},
            "transitions": [{"state": "HEALTHY", "reasons": ""}],
            "timing_ranges_ms": {
                "rgb_image.sensor_stamp_age_ms": {"maximum": 1.0}
            },
            "latest_volatile_values": {},
        }
        groups = [
            {
                "hazard_track_id": 2,
                "centroid_x_m": 0.5,
                "centroid_y_m": 0.0,
                "source_stamp_max_ns": stamp,
                "confirmed_detection_distance_m": 0.5,
                "confirmation_latency_ms": 100.0,
            }
            for stamp in (1_000_000_000, 1_300_000_000)
        ]
        run = {
            "health": health,
            "clouds": [{"clearing": False, "hazard_groups": groups}],
            "runtime": {
                "evaluated_duration_seconds": 1.0,
                "observed_maximum_speed_mps": 0.0,
                "resource_use": {
                    "processing_p95_ms": 1.0,
                    "depth_geometry_average_cpu_cores": 0.1,
                    "memory_growth_bytes": 0,
                    "maximum_pending_work": 0,
                },
            },
        }

        result = normalize_home_scene_replay(
            suite=suite,
            scene=scene,
            annotation=annotation,
            runs=[run, run],
            bag_sha256="b" * 64,
        )

        self.assertEqual(len(result["event_outcomes"]), 1)
        self.assertTrue(result["event_outcomes"][0]["detected"])

    def test_normal_run_cannot_be_relabelled_as_rgb_failure(self) -> None:
        run = {
            "health": {
                "canonical": {
                    "state": "DEGRADED",
                    "stable_values": {"reasons": "budget:processing"},
                },
                "transitions": [
                    {"state": "DEGRADED", "reasons": "budget:processing"}
                ],
                "latest_volatile_values": {},
            },
            "clouds": [],
        }

        outcomes = _failure_injection_outcomes(
            [
                {
                    "injection": "rgb",
                    "expected_health_state": "DEGRADED",
                    "forbid_new_confirmed_hazard": True,
                }
            ],
            {"rgb": run},
        )

        self.assertFalse(outcomes[0]["failure_observed"])
        self.assertFalse(outcomes[0]["passed"])


if __name__ == "__main__":
    unittest.main()
