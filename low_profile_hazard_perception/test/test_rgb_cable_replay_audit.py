import unittest

from low_profile_hazard_perception.cable_replay_audit import (
    audit_rgb_cable_replay,
)


class RgbCableReplayAuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
