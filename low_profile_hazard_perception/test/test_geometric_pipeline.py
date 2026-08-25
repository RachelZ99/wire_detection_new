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

    def test_reflective_floor_holes_do_not_confirm_as_geometry(self) -> None:
        depth, intrinsics = _scene(raised=False, reflective_hole=True)
        pipeline = _pipeline(intrinsics)

        first = pipeline.process_depth(1_000_000_000, depth)
        second = pipeline.process_depth(1_100_000_000, depth)

        assert first is not None and second is not None
        self.assertTrue(first.ground.accepted, first.ground.reason)
        self.assertEqual(first.candidates, ())
        self.assertEqual(second.confirmed, ())


if __name__ == "__main__":
    unittest.main()
