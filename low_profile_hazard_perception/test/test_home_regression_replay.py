import unittest

from low_profile_hazard_perception.home_regression_replay import (
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
                            "hazard_group_id": 0,
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


if __name__ == "__main__":
    unittest.main()
