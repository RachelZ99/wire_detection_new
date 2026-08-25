"""Depth-only strong-geometry path from one capture time into odom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import (
    CameraIntrinsics,
    GeometricCandidate,
    GroundEstimate,
    GroundEstimator,
    GroundEstimatorConfig,
    StrongGeometryConfig,
    StrongGeometryDetector,
)
from .temporal import (
    ConfirmedHazard,
    EvidenceSource,
    HazardObservation,
    HazardTracker,
    HazardTrackerConfig,
    OdomPoseCache,
    Pose3,
)


@dataclass(frozen=True)
class GeometricPipelineResult:
    sensor_stamp_ns: int
    ground: GroundEstimate
    candidates: tuple[GeometricCandidate, ...]
    confirmed: tuple[ConfirmedHazard, ...]
    nominal_ground_angle_error_degrees: float | None


class GeometricHazardPipeline:
    """Own the explicitly asynchronous geometry/alignment/confirmation path."""

    def __init__(
        self,
        *,
        ground_config: GroundEstimatorConfig | None = None,
        geometry_config: StrongGeometryConfig | None = None,
        tracker_config: HazardTrackerConfig | None = None,
    ) -> None:
        self.ground_estimator = GroundEstimator(ground_config)
        self.geometry_detector = StrongGeometryDetector(geometry_config)
        self.odom = OdomPoseCache()
        self.tracker = HazardTracker(tracker_config)
        self.intrinsics: CameraIntrinsics | None = None
        self.base_from_camera: Pose3 | None = None
        self.nominal_camera_height_m = 0.0
        self._last_aligned_observation_stamp_ns: int | None = None

    def set_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    def set_base_from_camera(self, pose: Pose3) -> None:
        self.base_from_camera = pose.normalized()
        self.nominal_camera_height_m = abs(pose.translation[2])

    def add_odom(self, sensor_stamp_ns: int, odom_from_base: Pose3) -> None:
        self.odom.add(sensor_stamp_ns, odom_from_base)

    def alignment_at(self, sensor_stamp_ns: int) -> Pose3 | None:
        if self.base_from_camera is None:
            return None
        odom_from_base = self.odom.interpolate(sensor_stamp_ns)
        if odom_from_base is None:
            return None
        return odom_from_base.compose(self.base_from_camera)

    def process_depth(
        self,
        sensor_stamp_ns: int,
        depth_values: Sequence[int | float],
        *,
        depth_unit_m: float = 0.001,
    ) -> GeometricPipelineResult | None:
        if self.intrinsics is None or self.base_from_camera is None:
            return None
        odom_from_base = self.odom.interpolate(sensor_stamp_ns)
        if odom_from_base is None:
            return None
        if (
            self._last_aligned_observation_stamp_ns is not None
            and not self.odom.continuous_between(
                self._last_aligned_observation_stamp_ns, sensor_stamp_ns
            )
        ):
            self.tracker.clear()
        self._last_aligned_observation_stamp_ns = sensor_stamp_ns
        ground = self.ground_estimator.estimate(
            depth_values,
            self.intrinsics,
            depth_unit_m=depth_unit_m,
            nominal_camera_height_m=self.nominal_camera_height_m,
        )
        nominal_angle_error = (
            self.base_from_camera.normal_error_to_parent_up_degrees(
                ground.model.normal
            )
        )
        if not ground.accepted:
            return GeometricPipelineResult(
                sensor_stamp_ns=sensor_stamp_ns,
                ground=ground,
                candidates=(),
                confirmed=(),
                nominal_ground_angle_error_degrees=nominal_angle_error,
            )
        observed_base_from_camera = self.base_from_camera.with_observed_ground(
            ground.model.normal, ground.model.camera_height_m
        )
        odom_from_camera = odom_from_base.compose(observed_base_from_camera)
        candidates = self.geometry_detector.detect(
            depth_values,
            self.intrinsics,
            ground.model,
            depth_unit_m=depth_unit_m,
        )
        confirmed: list[ConfirmedHazard] = []
        for candidate in candidates:
            points_odom = tuple(
                odom_from_camera.transform_point(point)
                for point in candidate.points
            )
            confirmed.extend(
                self.tracker.observe(
                    HazardObservation(
                        sensor_stamp_ns=sensor_stamp_ns,
                        points_odom=points_odom,
                        evidence=EvidenceSource.STRONG_GEOMETRY,
                        confidence=candidate.confidence,
                    )
                )
            )
        return GeometricPipelineResult(
            sensor_stamp_ns=sensor_stamp_ns,
            ground=ground,
            candidates=candidates,
            confirmed=tuple(confirmed),
            nominal_ground_angle_error_degrees=nominal_angle_error,
        )
