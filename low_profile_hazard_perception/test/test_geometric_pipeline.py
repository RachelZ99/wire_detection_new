import math
import unittest

from low_profile_hazard_perception.geometric_pipeline import (
    GeometricHazardPipeline,
)
from low_profile_hazard_perception.geometry import (
    CameraIntrinsics,
    GroundEstimatorConfig,
    StrongGeometryConfig,
)
from low_profile_hazard_perception.temporal import (
    CandidateDecisionReason,
    HazardTrackerConfig,
    Pose3,
)


def _scene(
    *, raised: bool, reflective_hole: bool
) -> tuple[list[int], CameraIntrinsics]:
    intrinsics = CameraIntrinsics(
        width=120, height=72, fx=100.0, fy=100.0, cx=60.0, cy=36.0
    )
    pitch = math.radians(3.0)
    normal = (0.0, -math.cos(pitch), -math.sin(pitch))
    offset = 0.225
    depth: list[int] = []
    for row in range(intrinsics.height):
        for column in range(intrinsics.width):
            ray = (
                (column - intrinsics.cx) / intrinsics.fx,
                (row - intrinsics.cy) / intrinsics.fy,
                1.0,
            )
            denominator = sum(
                left * right for left, right in zip(normal, ray, strict=True)
            )
            if denominator >= -1e-6:
                depth.append(0)
                continue
            height = (
                0.030
                if raised and 48 <= column < 76 and 52 <= row < 62
                else 0.0
            )
            value_m = (height - offset) / denominator
            depth.append(int(round(value_m * 1000.0)))
    if reflective_hole:
        for row in range(50, 66):
            for column in range(12, 42):
                depth[row * intrinsics.width + column] = 0
    return depth, intrinsics


def _pipeline(intrinsics: CameraIntrinsics) -> GeometricHazardPipeline:
    pipeline = GeometricHazardPipeline(
        ground_config=GroundEstimatorConfig(
            sample_stride_px=3,
            ransac_iterations=100,
            minimum_support=180,
            minimum_inlier_ratio=0.65,
            minimum_spatial_coverage=0.30,
        ),
        geometry_config=StrongGeometryConfig(
            sample_stride_px=1,
            minimum_support_points=24,
            cluster_cell_m=0.04,
        ),
        tracker_config=HazardTrackerConfig(
            association_radius_m=0.08,
            confirmation_window_ns=350_000_000,
        ),
    )
    pipeline.set_intrinsics(intrinsics)
    pipeline.set_base_from_camera(
        Pose3(
            translation=(0.33, 0.0, 0.15),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
    )
    for stamp in (950_000_000, 1_050_000_000, 1_150_000_000):
        pipeline.add_odom(stamp, Pose3.identity())
    return pipeline


class GeometricPipelineTests(unittest.TestCase):
    def test_two_supported_protrusions_confirm_but_first_does_not(
        self,
    ) -> None:
        depth, intrinsics = _scene(raised=True, reflective_hole=False)
        pipeline = _pipeline(intrinsics)

        first = pipeline.process_depth(1_000_000_000, depth)
        second = pipeline.process_depth(1_100_000_000, depth)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertTrue(first.ground.accepted, first.ground.reason)
        self.assertEqual(len(first.candidates), 1)
        self.assertEqual(first.confirmed, ())
        self.assertEqual(len(second.confirmed), 1)
        self.assertEqual(second.confirmed[0].sensor_stamp_ns, 1_100_000_000)
        heights = sorted(point[2] for point in second.confirmed[0].points_odom)
        self.assertAlmostEqual(heights[len(heights) // 2], 0.030, delta=0.004)

    def test_reflective_floor_holes_do_not_confirm_as_geometry(self) -> None:
        depth, intrinsics = _scene(raised=False, reflective_hole=True)
        pipeline = _pipeline(intrinsics)

        first = pipeline.process_depth(1_000_000_000, depth)
        second = pipeline.process_depth(1_100_000_000, depth)

        assert first is not None and second is not None
        self.assertTrue(first.ground.accepted, first.ground.reason)
        self.assertEqual(first.candidates, ())
        self.assertEqual(second.confirmed, ())
        self.assertTrue(first.candidate_reports)
        self.assertEqual(
            first.candidate_reports[0].decision_reason,
            CandidateDecisionReason.REJECTED_INVALID_DEPTH_TOO_WIDE,
        )

    def test_odom_outage_clears_cross_frame_confirmation(self) -> None:
        depth, intrinsics = _scene(raised=True, reflective_hole=False)
        pipeline = _pipeline(intrinsics)
        first = pipeline.process_depth(1_000_000_000, depth)
        assert first is not None
        self.assertEqual(first.confirmed, ())
        pipeline.add_odom(1_350_000_000, Pose3.identity())
        pipeline.add_odom(1_450_000_000, Pose3.identity())

        after_outage = pipeline.process_depth(1_400_000_000, depth)

        assert after_outage is not None
        self.assertEqual(len(after_outage.candidates), 1)
        self.assertEqual(after_outage.confirmed, ())
        self.assertEqual(after_outage.degradation_reason, "odom:discontinuous")

    def test_ground_failure_retains_a_confirmed_hazard(self) -> None:
        depth, intrinsics = _scene(raised=True, reflective_hole=False)
        pipeline = _pipeline(intrinsics)
        pipeline.add_odom(1_250_000_000, Pose3.identity())
        first = pipeline.process_depth(1_000_000_000, depth)
        second = pipeline.process_depth(1_100_000_000, depth)
        assert first is not None and second is not None
        self.assertEqual(len(second.confirmed), 1)

        failed = pipeline.process_depth(
            1_200_000_000,
            [0] * (intrinsics.width * intrinsics.height),
        )

        assert failed is not None
        self.assertFalse(failed.ground.accepted)
        self.assertEqual(failed.confirmed, ())
        self.assertEqual(len(failed.retained), 1)
        self.assertEqual(
            failed.retained[0].points_odom,
            second.confirmed[0].points_odom,
        )
        self.assertEqual(
            failed.retained[0].sensor_stamp_ns,
            second.confirmed[0].sensor_stamp_ns,
        )
        self.assertEqual(
            failed.degradation_reason,
            "ground:insufficient valid floor samples",
        )

        pipeline.add_odom(1_350_000_000, Pose3.identity())
        recovered = pipeline.process_depth(1_300_000_000, depth)

        assert recovered is not None
        self.assertTrue(recovered.ground.accepted, recovered.ground.reason)
        self.assertEqual(recovered.confirmed, ())
        self.assertEqual(len(recovered.retained), 1)
        self.assertEqual(recovered.retained[0].observation_count, 2)
        self.assertEqual(
            recovered.degradation_reason,
            "recovery:reconfirmation_required",
        )

        pipeline.add_odom(1_450_000_000, Pose3.identity())
        reconfirmed = pipeline.process_depth(1_400_000_000, depth)

        assert reconfirmed is not None
        self.assertEqual(len(reconfirmed.confirmed), 1)
        self.assertEqual(len(reconfirmed.retained), 1)
        self.assertEqual(reconfirmed.retained[0].observation_count, 4)
        self.assertEqual(reconfirmed.degradation_reason, "")

        pipeline.suspend_confirmed_expiry()
        self.assertEqual(len(pipeline.retained_at(4_000_000_000)), 1)

    def test_odom_discontinuity_retains_confirmed_but_restarts_candidates(
        self,
    ) -> None:
        depth, intrinsics = _scene(raised=True, reflective_hole=False)
        pipeline = _pipeline(intrinsics)
        first = pipeline.process_depth(1_000_000_000, depth)
        confirmed = pipeline.process_depth(1_100_000_000, depth)
        assert first is not None and confirmed is not None
        self.assertEqual(len(confirmed.confirmed), 1)
        pipeline.add_odom(1_350_000_000, Pose3.identity())
        pipeline.add_odom(1_450_000_000, Pose3.identity())

        recovered = pipeline.process_depth(1_400_000_000, depth)

        assert recovered is not None
        # A discontinuity preserves the operational shape but makes the first
        # post-gap observation diagnostic-only until a second one agrees.
        self.assertEqual(recovered.confirmed, ())
        self.assertEqual(len(recovered.retained), 1)
        self.assertEqual(recovered.retained[0].observation_count, 2)
        self.assertEqual(recovered.degradation_reason, "odom:discontinuous")

        pipeline.add_odom(1_550_000_000, Pose3.identity())
        reconfirmed = pipeline.process_depth(1_500_000_000, depth)

        assert reconfirmed is not None
        self.assertEqual(len(reconfirmed.confirmed), 1)
        self.assertEqual(len(reconfirmed.retained), 1)
        self.assertEqual(reconfirmed.degradation_reason, "")

    def test_disordered_odom_prevents_cross_frame_confirmation(self) -> None:
        depth, intrinsics = _scene(raised=True, reflective_hole=False)
        pipeline = _pipeline(intrinsics)
        first = pipeline.process_depth(1_000_000_000, depth)
        assert first is not None
        self.assertEqual(first.confirmed, ())

        reason = pipeline.add_odom(1_040_000_000, Pose3.identity())
        second = pipeline.process_depth(1_100_000_000, depth)

        self.assertEqual(reason, "odom:disordered")
        assert second is not None
        self.assertEqual(len(second.candidates), 1)
        self.assertEqual(second.confirmed, ())
        self.assertEqual(second.degradation_reason, "odom:disordered")


if __name__ == "__main__":
    unittest.main()
