import unittest

from low_profile_hazard_perception.resource_budget import (
    BudgetLimits,
    ResourceBudgetAudit,
    ResourceSample,
    StageMetrics,
)


class StageMetricsTests(unittest.TestCase):
    def test_stage_window_is_bounded_and_reports_processing_and_age(self) -> None:
        metrics = StageMetrics(capacity=3)
        for index, wall_ms in enumerate((10.0, 20.0, 30.0, 40.0)):
            metrics.record(
                started_monotonic_ns=index * 100_000_000,
                completed_monotonic_ns=index * 100_000_000
                + int(wall_ms * 1_000_000),
                cpu_time_ns=int(wall_ms * 0.5 * 1_000_000),
                queue_wait_ms=float(index),
                message_age_ms=100.0 + index,
            )

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot.sample_count, 4)
        self.assertEqual(snapshot.retained_sample_count, 3)
        self.assertEqual(snapshot.processing_wall_p95_ms, 40.0)
        self.assertEqual(snapshot.latest_message_age_ms, 103.0)
        self.assertEqual(snapshot.latest_queue_wait_ms, 3.0)
        self.assertLessEqual(snapshot.average_cpu_cores, 1.0)


class ResourceBudgetAuditTests(unittest.TestCase):
    def test_two_hour_stable_run_meets_budget(self) -> None:
        audit = ResourceBudgetAudit(
            BudgetLimits(
                processing_p95_ms=80.0,
                depth_geometry_average_cpu_cores=1.0,
                soak_duration_seconds=7200,
                maximum_memory_growth_bytes=32 * 1024 * 1024,
                maximum_pending_work=1,
            )
        )
        report = audit.evaluate(
            elapsed_seconds=7200.0,
            perception_processing_samples_ms=[50.0, 60.0, 70.0, 80.0],
            depth_geometry_average_cpu_cores=0.85,
            resource_samples=[
                ResourceSample(0.0, 120 * 1024 * 1024, 0.7),
                ResourceSample(3600.0, 124 * 1024 * 1024, 0.8),
                ResourceSample(7200.0, 122 * 1024 * 1024, 0.75),
            ],
            maximum_pending_work=1,
            frame_drops=4,
            npu_state="disabled_rule_profile",
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.processing_p95_ms, 80.0)
        self.assertEqual(report.memory_growth_bytes, 2 * 1024 * 1024)
        self.assertEqual(report.frame_drops, 4)

    def test_short_or_growing_run_is_an_evidence_backed_failure(self) -> None:
        audit = ResourceBudgetAudit(
            BudgetLimits(
                processing_p95_ms=80.0,
                depth_geometry_average_cpu_cores=1.0,
                soak_duration_seconds=7200,
                maximum_memory_growth_bytes=16 * 1024 * 1024,
                maximum_pending_work=1,
            )
        )
        report = audit.evaluate(
            elapsed_seconds=60.0,
            perception_processing_samples_ms=[50.0, 95.0],
            depth_geometry_average_cpu_cores=1.2,
            resource_samples=[
                ResourceSample(0.0, 100 * 1024 * 1024, 0.9),
                ResourceSample(60.0, 140 * 1024 * 1024, 1.2),
            ],
            maximum_pending_work=2,
            frame_drops=0,
            npu_state="unavailable",
        )

        self.assertFalse(report.passed)
        self.assertIn("processing_p95_exceeded", report.reasons)
        self.assertIn("depth_cpu_budget_exceeded", report.reasons)
        self.assertIn("soak_duration_incomplete", report.reasons)
        self.assertIn("memory_growth_exceeded", report.reasons)
        self.assertIn("pending_work_unbounded", report.reasons)


if __name__ == "__main__":
    unittest.main()
