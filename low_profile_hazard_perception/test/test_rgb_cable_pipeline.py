import math
import unittest

from low_profile_hazard_perception.cable import TrainingFreeCableConfig
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
    EvidenceSource,
    HazardTrackerConfig,
    Pose3,
)


def _floor_depth(intrinsics: CameraIntrinsics) -> list[int]:
    pitch = math.radians(3.0)
    normal = (0.0, -math.cos(pitch), -math.sin(pitch))
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
            depth.append(
                0 if denominator >= -1e-6 else int(round(-0.225 / denominator * 1000.0))
            )
    return depth


def _cable_rgb(intrinsics: CameraIntrinsics, center_column: int) -> bytes:
    data = bytearray((82, 86, 88) * (intrinsics.width * intrinsics.height))
    for row in range(48, 69):
        for column in range(center_column - 1, center_column + 2):
            offset = (row * intrinsics.width + column) * 3
            data[offset : offset + 3] = bytes((235, 235, 235))
    return bytes(data)


def _mixed_depth_evidence(intrinsics: CameraIntrinsics) -> list[int]:
    depth = _floor_depth(intrinsics)
    pitch = math.radians(3.0)
    normal = (0.0, -math.cos(pitch), -math.sin(pitch))
    for row in range(48, 69):
        for column in range(57, 60):
            ray = (
                (column - intrinsics.cx) / intrinsics.fx,
                (row - intrinsics.cy) / intrinsics.fy,
                1.0,
            )
            denominator = sum(
                left * right for left, right in zip(normal, ray, strict=True)
            )
            depth[row * intrinsics.width + column] = int(
                round((0.009 - 0.225) / denominator * 1000.0)
            )
        for column in range(60, 63):
            depth[row * intrinsics.width + column] = 0
    return depth


class RgbCablePipelineTests(unittest.TestCase):
    def test_mixed_pipeline_final_output_is_arrival_order_independent(self) -> None:
        intrinsics = CameraIntrinsics(
            width=120,
            height=72,
            fx=100.0,
            fy=100.0,
            cx=60.0,
            cy=36.0,
        )

        def make_pipeline() -> GeometricHazardPipeline:
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
                ),
                cable_config=TrainingFreeCableConfig(
                    minimum_component_pixels=12,
                    minimum_length_px=12.0,
                    minimum_physical_span_m=0.04,
                ),
            )
            pipeline.set_intrinsics(intrinsics)
            pipeline.set_rgb_intrinsics(intrinsics)
            pipeline.set_base_from_camera(
                Pose3((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0))
            )
            for stamp_ns in range(
                850_000_000, 1_351_000_000, 50_000_000
            ):
                pipeline.add_odom(stamp_ns, Pose3.identity())
            initial = pipeline.process_depth(
                900_000_000, _floor_depth(intrinsics)
            )
            assert initial is not None and initial.ground.accepted
            return pipeline

        outcomes = []
        for order in (
            ("depth", "early_rgb", "late_rgb"),
            ("late_rgb", "early_rgb", "depth"),
        ):
            pipeline = make_pipeline()
            for source in order:
                if source == "depth":
                    pipeline.process_depth(
                        1_050_000_000, _mixed_depth_evidence(intrinsics)
                    )
                elif source == "early_rgb":
                    pipeline.process_rgb(
                        1_000_000_000, _cable_rgb(intrinsics, 60)
                    )
                else:
                    pipeline.process_rgb(
                        1_200_000_000, _cable_rgb(intrinsics, 60)
                    )
            outcomes.append(pipeline.retained_at(1_200_000_000))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(len(outcomes[0]), 1)
        self.assertEqual(
            outcomes[0][0].evidence,
            (
                EvidenceSource.INVALID_DEPTH,
                EvidenceSource.RGB_CABLE,
                EvidenceSource.WEAK_HEIGHT,
            ),
        )

    def test_weak_height_and_continuous_invalid_depth_strengthen_rgb(self) -> None:
        intrinsics = CameraIntrinsics(
            width=120,
            height=72,
            fx=100.0,
            fy=100.0,
            cx=60.0,
            cy=36.0,
        )
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
            ),
            cable_config=TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            ),
        )
        pipeline.set_intrinsics(intrinsics)
        pipeline.set_rgb_intrinsics(intrinsics)
        pipeline.set_base_from_camera(Pose3((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)))
        for stamp_ns in range(950_000_000, 1_351_000_000, 50_000_000):
            pipeline.add_odom(stamp_ns, Pose3.identity())

        depth = pipeline.process_depth(1_050_000_000, _mixed_depth_evidence(intrinsics))
        first_rgb = pipeline.process_rgb(1_000_000_000, _cable_rgb(intrinsics, 60))
        second_rgb = pipeline.process_rgb(1_200_000_000, _cable_rgb(intrinsics, 60))

        assert depth is not None and first_rgb is not None
        assert second_rgb is not None
        self.assertEqual(depth.candidates, ())
        self.assertTrue(
            any(
                report.decision_reason is CandidateDecisionReason.SUPPORT_ONLY
                for report in depth.candidate_reports
            )
        )
        self.assertEqual(len(second_rgb.confirmed), 1)
        self.assertEqual(
            second_rgb.confirmed[0].evidence,
            (
                EvidenceSource.INVALID_DEPTH,
                EvidenceSource.RGB_CABLE,
                EvidenceSource.WEAK_HEIGHT,
            ),
        )
        report = second_rgb.candidate_reports[0]
        self.assertTrue(report.ground_accepted)
        self.assertGreater(report.ground_quality.inlier_ratio, 0.65)
        self.assertGreater(report.confidence, 0.85)
        self.assertEqual(
            report.decision_reason,
            CandidateDecisionReason.CONFIRMED_MIXED_EVIDENCE,
        )

    def test_rejected_ground_revokes_rgb_projection_support(self) -> None:
        intrinsics = CameraIntrinsics(
            width=120,
            height=72,
            fx=100.0,
            fy=100.0,
            cx=60.0,
            cy=36.0,
        )
        pipeline = GeometricHazardPipeline(
            ground_config=GroundEstimatorConfig(
                sample_stride_px=3,
                ransac_iterations=100,
                minimum_support=180,
                minimum_inlier_ratio=0.65,
                minimum_spatial_coverage=0.30,
            ),
            cable_config=TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            ),
        )
        pipeline.set_intrinsics(intrinsics)
        pipeline.set_rgb_intrinsics(intrinsics)
        pipeline.set_base_from_camera(Pose3((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)))
        for stamp_ns in range(950_000_000, 1_351_000_000, 100_000_000):
            pipeline.add_odom(stamp_ns, Pose3.identity())
        accepted = pipeline.process_depth(
            1_000_000_000,
            _floor_depth(intrinsics),
        )
        assert accepted is not None
        rejected = pipeline.process_depth(
            1_100_000_000,
            [0] * (intrinsics.width * intrinsics.height),
        )
        assert rejected is not None
        self.assertFalse(rejected.ground.accepted)

        unsupported = pipeline.process_rgb(
            1_200_000_000,
            _cable_rgb(intrinsics, 60),
        )

        self.assertIsNone(unsupported)

    def test_stale_observed_floor_cannot_project_new_rgb_evidence(self) -> None:
        intrinsics = CameraIntrinsics(
            width=120,
            height=72,
            fx=100.0,
            fy=100.0,
            cx=60.0,
            cy=36.0,
        )
        pipeline = GeometricHazardPipeline(
            ground_config=GroundEstimatorConfig(
                sample_stride_px=3,
                ransac_iterations=100,
                minimum_support=180,
                minimum_inlier_ratio=0.65,
                minimum_spatial_coverage=0.30,
            ),
            cable_config=TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
                maximum_ground_age_ns=500_000_000,
            ),
        )
        pipeline.set_intrinsics(intrinsics)
        pipeline.set_rgb_intrinsics(intrinsics)
        pipeline.set_base_from_camera(Pose3((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)))
        for stamp_ns in range(950_000_000, 1_751_000_000, 100_000_000):
            pipeline.add_odom(stamp_ns, Pose3.identity())
        ground_result = pipeline.process_depth(
            1_000_000_000,
            _floor_depth(intrinsics),
        )
        assert ground_result is not None

        stale = pipeline.process_rgb(
            1_600_000_000,
            _cable_rgb(intrinsics, 60),
        )

        self.assertIsNotNone(stale)
        assert stale is not None
        self.assertEqual(stale.candidates, ())
        self.assertEqual(stale.confirmed, ())
        self.assertEqual(stale.degradation_reason, "ground:stale")

    def test_independently_stamped_rgb_observations_confirm_in_odom(self) -> None:
        intrinsics = CameraIntrinsics(
            width=120,
            height=72,
            fx=100.0,
            fy=100.0,
            cx=60.0,
            cy=36.0,
        )
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
            ),
            cable_config=TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            ),
            tracker_config=HazardTrackerConfig(
                association_radius_m=0.08,
                confirmation_window_ns=350_000_000,
            ),
        )
        pipeline.set_intrinsics(intrinsics)
        pipeline.set_rgb_intrinsics(intrinsics)
        pipeline.set_base_from_camera(
            Pose3(
                translation=(0.0, 0.0, 0.15),
                rotation=(0.0, 0.0, 0.0, 1.0),
            )
        )
        for stamp_ns, x in (
            (950_000_000, 0.0),
            (1_050_000_000, 0.0),
            (1_150_000_000, 0.10),
            (1_250_000_000, 0.10),
        ):
            pipeline.add_odom(
                stamp_ns,
                Pose3(
                    translation=(x, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ground_result = pipeline.process_depth(
            1_000_000_000,
            _floor_depth(intrinsics),
        )
        assert ground_result is not None
        self.assertTrue(ground_result.ground.accepted)

        first = pipeline.process_rgb(
            1_000_000_000,
            _cable_rgb(intrinsics, 60),
        )
        second = pipeline.process_rgb(
            1_200_000_000,
            _cable_rgb(intrinsics, 47),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(first.confirmed, ())
        self.assertEqual(len(first.candidates), 1)
        self.assertEqual(len(second.confirmed), 1)
        confirmed = second.confirmed[0]
        self.assertEqual(confirmed.sensor_stamp_ns, 1_200_000_000)
        self.assertIn(EvidenceSource.RGB_CABLE, confirmed.evidence)
        self.assertLess(confirmed.spatial_spread_m, 0.03)
        self.assertEqual(len(second.retained), 1)


if __name__ == "__main__":
    unittest.main()
