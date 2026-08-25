import unittest

from low_profile_hazard_perception.cable_replay_audit import (
    audit_rgb_cable_replay,
)


class RgbCableReplayAuditTests(unittest.TestCase):
    def test_negative_only_replay_passes_without_a_confirmed_cable(self) -> None:
        audit = audit_rgb_cable_replay(
            stable_values={
                "cable.provider": "training_free_thin_line",
                "cable.confirmed_observation_count": "0",
                "cable.confirmation_observations": "2",
                "cable.processed_rgb_count": "8",
                "cable.rgb_depth_synchronizer": "disabled",
                "cable.diagnostic_pink_operational": "false",
            },
            clouds=[],
            maximum_alignment_spread_m=0.025,
            minimum_physical_span_m=0.06,
            negative_regions=(
                ("empty_reflective_floor", 0.4, 1.2, -0.3, 0.3),
            ),
            require_positive=False,
        )

        self.assertEqual(audit["confirmed_cable_event_count"], 0)
        self.assertEqual(audit["supported_odom_cloud_count"], 0)

    def test_requires_training_free_confirmation_and_an_aligned_odom_cloud(
        self,
    ) -> None:
        audit = audit_rgb_cable_replay(
            stable_values={
                "cable.provider": "training_free_thin_line",
                "cable.confirmed_observation_count": "1",
                "cable.confirmation_observations": "2",
                "cable.processed_rgb_count": "8",
                "cable.rgb_depth_synchronizer": "disabled",
                "cable.diagnostic_pink_operational": "false",
            },
            clouds=[
                {
                    "clearing": False,
                    "frame_id": "odom",
                    "stamp_ns": 1_200_000_000,
                    "source_stamp_min_ns": 1_200_000_000,
                    "confirmation_spread_m": 0.012,
                    "horizontal_span_m": 0.32,
                    "rgb_cable_point_count": 64,
                    "rgb_cable_span_m": 0.30,
                    "rgb_cable_centroid_x_m": 0.9,
                    "rgb_cable_centroid_y_m": 0.1,
                    "centroid_x_m": 0.9,
                    "centroid_y_m": 0.1,
                }
            ],
            maximum_alignment_spread_m=0.025,
            minimum_physical_span_m=0.06,
        )

        self.assertEqual(audit["confirmed_cable_event_count"], 1)
        self.assertEqual(audit["processed_rgb_count"], 8)

    def test_diagnostic_pink_count_cannot_satisfy_formal_replay(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-free"):
            audit_rgb_cable_replay(
                stable_values={
                    "cable.provider": "diagnostic_pink",
                    "cable.confirmed_observation_count": "0",
                    "cable.confirmation_observations": "2",
                    "cable.processed_rgb_count": "8",
                    "cable.rgb_depth_synchronizer": "disabled",
                    "cable.diagnostic_pink_operational": "false",
                    "cable.diagnostic_pink_pixel_count": "1200",
                },
                clouds=[],
                maximum_alignment_spread_m=0.025,
                minimum_physical_span_m=0.06,
            )

    def test_rejects_persistent_cable_evidence_in_named_negative_region(
        self,
    ) -> None:
        values = {
            "cable.provider": "training_free_thin_line",
            "cable.confirmed_observation_count": "1",
            "cable.confirmation_observations": "2",
            "cable.processed_rgb_count": "8",
            "cable.rgb_depth_synchronizer": "disabled",
            "cable.diagnostic_pink_operational": "false",
        }
        cloud = {
            "clearing": False,
            "frame_id": "odom",
            "stamp_ns": 1_200_000_000,
            "source_stamp_min_ns": 1_200_000_000,
            "confirmation_spread_m": 0.012,
            "rgb_cable_point_count": 64,
            "rgb_cable_span_m": 0.30,
            "rgb_cable_centroid_x_m": 0.9,
            "rgb_cable_centroid_y_m": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "long_shadow"):
            audit_rgb_cable_replay(
                stable_values=values,
                clouds=[cloud, {**cloud, "stamp_ns": 1_300_000_000, "source_stamp_min_ns": 1_300_000_000}],
                maximum_alignment_spread_m=0.025,
                minimum_physical_span_m=0.06,
                negative_regions=(
                    ("long_shadow", 0.8, 1.0, 0.0, 0.2),
                ),
            )


if __name__ == "__main__":
    unittest.main()
