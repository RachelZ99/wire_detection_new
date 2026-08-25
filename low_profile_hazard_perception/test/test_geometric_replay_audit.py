import unittest

from low_profile_hazard_perception.geometric_replay_audit import (
    observation_blind_zone_retention_audit,
)


class GeometricReplayAuditTests(unittest.TestCase):
    def test_audits_observation_blind_zone_retention_with_later_depth(
        self,
    ) -> None:
        audit = observation_blind_zone_retention_audit(
            [
                {
                    "clearing": False,
                    "source_stamp_max_ns": 1_100_000_000,
                    "stamp_ns": 1_100_000_000,
                },
                {
                    "clearing": True,
                    "source_stamp_max_ns": None,
                    "stamp_ns": 3_100_000_000,
                },
            ],
            latest_processed_depth_stamp_ns=2_900_000_000,
            minimum_retention_ns=2_000_000_000,
        )

        self.assertEqual(audit["retention_ns"], 2_000_000_000)
        self.assertEqual(
            audit["depth_after_observation_blind_zone_entry_ns"],
            1_800_000_000,
        )

    def test_rejects_clear_before_the_conservative_retention_interval(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "shorter"):
            observation_blind_zone_retention_audit(
                [
                    {
                        "clearing": False,
                        "source_stamp_max_ns": 1_100_000_000,
                        "stamp_ns": 1_100_000_000,
                    },
                    {
                        "clearing": True,
                        "source_stamp_max_ns": None,
                        "stamp_ns": 3_099_999_999,
                    },
                ],
                latest_processed_depth_stamp_ns=2_900_000_000,
                minimum_retention_ns=2_000_000_000,
            )

    def test_rejects_no_depth_after_observation_blind_zone_entry(self) -> None:
        with self.assertRaisesRegex(ValueError, "continued depth"):
            observation_blind_zone_retention_audit(
                [
                    {
                        "clearing": False,
                        "source_stamp_max_ns": 1_100_000_000,
                        "stamp_ns": 1_100_000_000,
                    },
                    {
                        "clearing": True,
                        "source_stamp_max_ns": None,
                        "stamp_ns": 3_100_000_000,
                    },
                ],
                latest_processed_depth_stamp_ns=1_100_000_000,
                minimum_retention_ns=2_000_000_000,
            )


if __name__ == "__main__":
    unittest.main()
