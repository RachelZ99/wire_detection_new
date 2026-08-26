import unittest

from low_profile_hazard_perception.replay_result import ReplayResultAccumulator


class ReplayResultTests(unittest.TestCase):
    def test_records_transitions_and_timing_without_canonicalizing_host_time(
        self,
    ) -> None:
        result = ReplayResultAccumulator(
            diagnostic_name="low_profile_hazard_perception/input_health"
        )
        result.record(
            state="INVALID",
            values={
                "state": "INVALID",
                "reasons": "invalid:depth_image",
                "depth_image.delivered_count": "1",
                "depth_image.sensor_stamp_age_ms": "20.000",
                "depth_image.receive_age_ms": "4.000",
                "depth_image.processing_latency_ms": "2.000",
                "stage.perception.processing_wall_p95_ms": "55.000",
                "resource.memory_rss_bytes": "125829120",
            },
        )
        result.record(
            state="HEALTHY",
            values={
                "state": "HEALTHY",
                "reasons": "",
                "depth_image.delivered_count": "2",
                "depth_image.sensor_stamp_age_ms": "10.000",
                "depth_image.receive_age_ms": "3.000",
                "depth_image.processing_latency_ms": "1.000",
                "stage.perception.processing_wall_p95_ms": "54.000",
                "resource.memory_rss_bytes": "126877696",
            },
        )

        report = result.report()

        self.assertTrue(report["invalid_observed"])
        self.assertEqual(
            [item["state"] for item in report["transitions"]],
            ["INVALID", "HEALTHY"],
        )
        self.assertEqual(
            report["timing_ranges_ms"]["depth_image.receive_age_ms"],
            {"minimum": 3.0, "maximum": 4.0, "last": 3.0},
        )
        self.assertNotIn(
            "depth_image.receive_age_ms",
            report["canonical"]["stable_values"],
        )
        self.assertNotIn(
            "stage.perception.processing_wall_p95_ms",
            report["canonical"]["stable_values"],
        )
        self.assertNotIn(
            "resource.memory_rss_bytes",
            report["canonical"]["stable_values"],
        )
        self.assertEqual(
            report["canonical"]["stable_values"][
                "depth_image.delivered_count"
            ],
            "2",
        )


if __name__ == "__main__":
    unittest.main()
