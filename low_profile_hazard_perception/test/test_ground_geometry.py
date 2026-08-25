import math
import unittest

from low_profile_hazard_perception.geometry import (
    CameraIntrinsics,
    GroundEstimator,
    GroundEstimatorConfig,
    StrongGeometryConfig,
    StrongGeometryDetector,
)


def _plane_depth_image(
    *,
    width: int,
    height: int,
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    downward_pitch_degrees: float,
) -> list[int]:
    pitch = math.radians(downward_pitch_degrees)
    normal = (0.0, -math.cos(pitch), -math.sin(pitch))
    values: list[int] = []
    for row in range(height):
        ray_y = (row - intrinsics.cy) / intrinsics.fy
        for column in range(width):
            ray_x = (column - intrinsics.cx) / intrinsics.fx
            denominator = normal[0] * ray_x + normal[1] * ray_y + normal[2]
            if denominator >= -1e-6:
                values.append(0)
                continue
            depth_m = -camera_height_m / denominator
            values.append(int(round(depth_m * 1000.0)))
    return values


class ObservedGroundModelTests(unittest.TestCase):
    def test_ambiguous_or_unobserved_ground_is_rejected_deterministically(
        self,
    ) -> None:
        intrinsics = CameraIntrinsics(
            width=160,
            height=90,
            fx=114.0,
            fy=114.0,
            cx=80.0,
            cy=45.0,
        )
        first_plane = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.225,
            downward_pitch_degrees=2.7,
        )
        second_plane = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.300,
            downward_pitch_degrees=2.7,
        )
        estimator = GroundEstimator(
            GroundEstimatorConfig(
                sample_stride_px=2,
                minimum_support=100,
                minimum_inlier_ratio=0.70,
                minimum_spatial_coverage=0.35,
            )
        )

        competing = [
            second if (index % intrinsics.width) // 10 % 2 else first
            for index, (first, second) in enumerate(
                zip(first_plane, second_plane, strict=True)
            )
        ]
        competing_result = estimator.estimate(
            competing,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertFalse(competing_result.accepted)
        self.assertEqual(
            competing_result.reason, "ground inlier ratio is too low"
        )

        narrow_floor = list(first_plane)
        for row in range(intrinsics.height):
            for column in range(intrinsics.width):
                if not 70 <= column < 90:
                    narrow_floor[row * intrinsics.width + column] = 0
        coverage_result = estimator.estimate(
            narrow_floor,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertFalse(coverage_result.accepted)
        self.assertEqual(
            coverage_result.reason, "ground spatial coverage is too small"
        )

        invalid_depth_result = estimator.estimate(
            [0] * (intrinsics.width * intrinsics.height),
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertFalse(invalid_depth_result.accepted)
        self.assertEqual(
            invalid_depth_result.reason,
            "insufficient valid floor samples",
        )

    def test_valid_depth_reports_measured_ground_quality_not_nominal_height(
        self,
    ) -> None:
        intrinsics = CameraIntrinsics(
            width=160,
            height=90,
            fx=114.0,
            fy=114.0,
            cx=80.0,
            cy=45.0,
        )
        depth_mm = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.225,
            downward_pitch_degrees=2.7,
        )
        estimator = GroundEstimator(
            GroundEstimatorConfig(
                sample_stride_px=3,
                minimum_support=250,
                minimum_inlier_ratio=0.7,
                minimum_spatial_coverage=0.35,
            )
        )

        estimate = estimator.estimate(
            depth_mm,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )

        self.assertTrue(estimate.accepted, estimate.reason)
        self.assertGreater(estimate.metrics.support_count, 250)
        self.assertGreater(estimate.metrics.inlier_ratio, 0.95)
        self.assertLess(estimate.metrics.p90_residual_m, 0.002)
        self.assertGreater(estimate.metrics.spatial_coverage, 0.35)
        self.assertGreater(estimate.metrics.temporal_consistency, 0.99)
        self.assertAlmostEqual(
            estimate.model.camera_height_m, 0.225, delta=0.004
        )
        self.assertGreater(estimate.metrics.nominal_height_error_m, 0.06)

        shifted_depth = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.235,
            downward_pitch_degrees=2.7,
        )
        shifted = estimator.estimate(
            shifted_depth,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertTrue(shifted.accepted, shifted.reason)
        self.assertAlmostEqual(
            shifted.metrics.height_change_m, 0.010, delta=0.002
        )
        self.assertGreater(shifted.metrics.temporal_consistency, 0.4)
        self.assertLess(shifted.metrics.temporal_consistency, 0.8)
        # The accepted model moves through the configured short-term smoother.
        self.assertGreater(shifted.model.camera_height_m, 0.225)
        self.assertLess(shifted.model.camera_height_m, 0.235)

        inconsistent_depth = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.300,
            downward_pitch_degrees=12.0,
        )
        inconsistent = estimator.estimate(
            inconsistent_depth,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertFalse(inconsistent.accepted)
        self.assertIn("temporal consistency", inconsistent.reason)

    def test_strong_geometry_requires_robust_local_height_support(
        self,
    ) -> None:
        intrinsics = CameraIntrinsics(
            width=160,
            height=90,
            fx=114.0,
            fy=114.0,
            cx=80.0,
            cy=45.0,
        )
        depth_mm = _plane_depth_image(
            width=160,
            height=90,
            intrinsics=intrinsics,
            camera_height_m=0.225,
            downward_pitch_degrees=2.7,
        )
        estimator = GroundEstimator(
            GroundEstimatorConfig(
                sample_stride_px=3,
                minimum_support=250,
                minimum_inlier_ratio=0.7,
                minimum_spatial_coverage=0.35,
            )
        )
        estimate = estimator.estimate(
            depth_mm,
            intrinsics,
            depth_unit_m=0.001,
            nominal_camera_height_m=0.15,
        )
        self.assertTrue(estimate.accepted, estimate.reason)

        normal = estimate.model.normal
        for row in range(63, 73):
            for column in range(65, 95):
                ray = (
                    (column - intrinsics.cx) / intrinsics.fx,
                    (row - intrinsics.cy) / intrinsics.fy,
                    1.0,
                )
                normal_dot_ray = sum(
                    left * right
                    for left, right in zip(normal, ray, strict=True)
                )
                depth_m = (0.030 - estimate.model.offset_m) / normal_dot_ray
                depth_mm[row * intrinsics.width + column] = int(
                    round(depth_m * 1000.0)
                )
        # One much taller depth point must not become an obstacle by itself.
        depth_mm[75 * intrinsics.width + 120] -= 100

        detector = StrongGeometryDetector(
            StrongGeometryConfig(
                sample_stride_px=1,
                minimum_support_points=30,
                cluster_cell_m=0.03,
            )
        )
        candidates = detector.detect(
            depth_mm,
            intrinsics,
            estimate.model,
            depth_unit_m=0.001,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertGreater(candidate.support_count, 200)
        self.assertGreater(candidate.p20_height_m, 0.025)
        self.assertAlmostEqual(candidate.p90_height_m, 0.030, delta=0.004)
        self.assertGreater(candidate.spatial_span_m, 0.10)


if __name__ == "__main__":
    unittest.main()
