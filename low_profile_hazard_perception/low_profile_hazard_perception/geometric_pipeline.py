"""Independent depth geometry and RGB cable paths carried into odom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cable import (
    CableCandidate,
    DiagnosticPinkConfig,
    ObservedFloorRegion,
    TrainingFreeCableConfig,
    TrainingFreeCableDetector,
    diagnostic_pink_pixel_count,
)
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
    retained: tuple[ConfirmedHazard, ...]
    nominal_ground_angle_error_degrees: float | None
    degradation_reason: str = ""


@dataclass(frozen=True)
class RgbCablePipelineResult:
    sensor_stamp_ns: int
    ground_sensor_stamp_ns: int
    candidates: tuple[CableCandidate, ...]
    confirmed: tuple[ConfirmedHazard, ...]
    retained: tuple[ConfirmedHazard, ...]
    degradation_reason: str = ""


class GeometricHazardPipeline:
    """Own the explicitly asynchronous geometry/alignment/confirmation path."""

    def __init__(
        self,
        *,
        ground_config: GroundEstimatorConfig | None = None,
        geometry_config: StrongGeometryConfig | None = None,
        cable_config: TrainingFreeCableConfig | None = None,
        tracker_config: HazardTrackerConfig | None = None,
    ) -> None:
        self.ground_estimator = GroundEstimator(ground_config)
        self.geometry_detector = StrongGeometryDetector(geometry_config)
        self.cable_detector = TrainingFreeCableDetector(cable_config)
        self.odom = OdomPoseCache()
        self.tracker = HazardTracker(tracker_config)
        self.intrinsics: CameraIntrinsics | None = None
        self.rgb_intrinsics: CameraIntrinsics | None = None
        self.base_from_camera: Pose3 | None = None
        self.nominal_camera_height_m = 0.0
        self._last_aligned_observation_stamp_ns: int | None = None
        self._last_aligned_rgb_stamp_ns: int | None = None
        self._latest_floor: tuple[
            int, GroundEstimate, ObservedFloorRegion
        ] | None = None
        self._pending_alignment_reason = ""
        self._confirmed_expiry_suspended = False

    def set_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    def set_rgb_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        self.rgb_intrinsics = intrinsics

    def set_base_from_camera(self, pose: Pose3) -> None:
        self.base_from_camera = pose.normalized()
        self.nominal_camera_height_m = abs(pose.translation[2])

    @property
    def alignment_reason(self) -> str:
        return self._pending_alignment_reason

    def add_odom(self, sensor_stamp_ns: int, odom_from_base: Pose3) -> str:
        reason = self.odom.add(sensor_stamp_ns, odom_from_base)
        if reason:
            self._suspend_confirmation()
            self._pending_alignment_reason = reason
        return reason

    def retained_at(
        self, sensor_now_ns: int
    ) -> tuple[ConfirmedHazard, ...]:
        return self.tracker.retained_at(
            sensor_now_ns,
            allow_confirmed_expiry=not self._confirmed_expiry_suspended,
        )

    def suspend_confirmed_expiry(self) -> None:
        self._suspend_confirmation()

    def _suspend_confirmation(self) -> None:
        self.tracker.clear_candidates()
        self._confirmed_expiry_suspended = True

    def alignment_at(self, sensor_stamp_ns: int) -> Pose3 | None:
        if self.base_from_camera is None:
            return None
        alignment = self.odom.alignment_at(sensor_stamp_ns)
        if alignment.pose is None:
            self._suspend_confirmation()
            self._pending_alignment_reason = alignment.reason
            return None
        return alignment.pose.compose(self.base_from_camera)

    def process_depth(
        self,
        sensor_stamp_ns: int,
        depth_values: Sequence[int | float],
        *,
        depth_unit_m: float = 0.001,
    ) -> GeometricPipelineResult | None:
        if self.intrinsics is None or self.base_from_camera is None:
            return None
        alignment = self.odom.alignment_at(sensor_stamp_ns)
        if alignment.pose is None:
            self._suspend_confirmation()
            self._pending_alignment_reason = alignment.reason
            return None
        odom_from_base = alignment.pose
        degradation_reason = self._pending_alignment_reason
        if degradation_reason:
            self._suspend_confirmation()
            self._pending_alignment_reason = ""
        if (
            self._last_aligned_observation_stamp_ns is not None
            and not self.odom.continuous_between(
                self._last_aligned_observation_stamp_ns, sensor_stamp_ns
            )
        ):
            self._suspend_confirmation()
            degradation_reason = "odom:discontinuous"
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
            self._suspend_confirmation()
            self._latest_floor = None
            return GeometricPipelineResult(
                sensor_stamp_ns=sensor_stamp_ns,
                ground=ground,
                candidates=(),
                confirmed=(),
                retained=self.tracker.retained_at(
                    sensor_stamp_ns,
                    allow_confirmed_expiry=False,
                ),
                nominal_ground_angle_error_degrees=nominal_angle_error,
                degradation_reason=f"ground:{ground.reason}",
            )
        observed_base_from_camera = self.base_from_camera.with_observed_ground(
            ground.model.normal, ground.model.camera_height_m
        )
        self._latest_floor = (
            sensor_stamp_ns,
            ground,
            ObservedFloorRegion.from_depth(
                depth_values,
                self.intrinsics,
                ground.model,
                depth_unit_m=depth_unit_m,
            ),
        )
        odom_from_camera = odom_from_base.compose(observed_base_from_camera)
        candidates = self.geometry_detector.detect(
            depth_values,
            self.intrinsics,
            ground.model,
            depth_unit_m=depth_unit_m,
        )
        confirmed: list[ConfirmedHazard] = []
        reconfirmation_required = self._confirmed_expiry_suspended
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
                    ),
                    allow_confirmed_expiry=(
                        not self._confirmed_expiry_suspended
                    ),
                    require_reconfirmation_for_confirmed=(
                        reconfirmation_required
                    ),
                )
            )
        if confirmed and not degradation_reason:
            self._confirmed_expiry_suspended = False
        elif reconfirmation_required and not degradation_reason:
            degradation_reason = "recovery:reconfirmation_required"
        return GeometricPipelineResult(
            sensor_stamp_ns=sensor_stamp_ns,
            ground=ground,
            candidates=candidates,
            confirmed=tuple(confirmed),
            retained=self.tracker.retained_at(
                sensor_stamp_ns,
                allow_confirmed_expiry=(
                    not self._confirmed_expiry_suspended
                ),
            ),
            nominal_ground_angle_error_degrees=nominal_angle_error,
            degradation_reason=degradation_reason,
        )

    def process_rgb(
        self,
        sensor_stamp_ns: int,
        rgb_values: bytes | bytearray | memoryview,
    ) -> RgbCablePipelineResult | None:
        """Process RGB at its own sensor timestamp without image pairing."""
        if (
            self.rgb_intrinsics is None
            or self.base_from_camera is None
            or self._latest_floor is None
        ):
            return None
        alignment = self.odom.alignment_at(sensor_stamp_ns)
        if alignment.pose is None:
            self._suspend_confirmation()
            self._pending_alignment_reason = alignment.reason
            return None
        degradation_reason = self._pending_alignment_reason
        if degradation_reason:
            self._suspend_confirmation()
            self._pending_alignment_reason = ""
        if (
            self._last_aligned_rgb_stamp_ns is not None
            and not self.odom.continuous_between(
                self._last_aligned_rgb_stamp_ns,
                sensor_stamp_ns,
            )
        ):
            self._suspend_confirmation()
            degradation_reason = "odom:discontinuous"
        self._last_aligned_rgb_stamp_ns = sensor_stamp_ns
        ground_stamp_ns, ground, floor_region = self._latest_floor
        if (
            abs(sensor_stamp_ns - ground_stamp_ns)
            > self.cable_detector.config.maximum_ground_age_ns
        ):
            self._suspend_confirmation()
            return RgbCablePipelineResult(
                sensor_stamp_ns=sensor_stamp_ns,
                ground_sensor_stamp_ns=ground_stamp_ns,
                candidates=(),
                confirmed=(),
                retained=self.tracker.retained_at(
                    sensor_stamp_ns,
                    allow_confirmed_expiry=False,
                ),
                degradation_reason="ground:stale",
            )
        candidates = self.cable_detector.detect(
            rgb_values,
            self.rgb_intrinsics,
            ground.model,
            floor_region,
        )
        observed_base_from_camera = self.base_from_camera.with_observed_ground(
            ground.model.normal,
            ground.model.camera_height_m,
        )
        odom_from_camera = alignment.pose.compose(observed_base_from_camera)
        confirmed: list[ConfirmedHazard] = []
        reconfirmation_required = self._confirmed_expiry_suspended
        for candidate in candidates:
            confirmed.extend(
                self.tracker.observe(
                    HazardObservation(
                        sensor_stamp_ns=sensor_stamp_ns,
                        points_odom=tuple(
                            odom_from_camera.transform_point(point)
                            for point in candidate.points_camera
                        ),
                        evidence=EvidenceSource.RGB_CABLE,
                        confidence=candidate.confidence,
                    ),
                    allow_confirmed_expiry=(
                        not self._confirmed_expiry_suspended
                    ),
                    require_reconfirmation_for_confirmed=(
                        reconfirmation_required
                    ),
                )
            )
        if confirmed and not degradation_reason:
            self._confirmed_expiry_suspended = False
        elif reconfirmation_required and not degradation_reason:
            degradation_reason = "recovery:reconfirmation_required"
        return RgbCablePipelineResult(
            sensor_stamp_ns=sensor_stamp_ns,
            ground_sensor_stamp_ns=ground_stamp_ns,
            candidates=candidates,
            confirmed=tuple(confirmed),
            retained=self.tracker.retained_at(
                sensor_stamp_ns,
                allow_confirmed_expiry=(
                    not self._confirmed_expiry_suspended
                ),
            ),
            degradation_reason=degradation_reason,
        )

    def diagnostic_pink_count(
        self,
        rgb_values: bytes | bytearray | memoryview,
        config: DiagnosticPinkConfig | None = None,
    ) -> int | None:
        """Run the color demo comparison without observing the tracker."""
        if self.rgb_intrinsics is None or self._latest_floor is None:
            return None
        return diagnostic_pink_pixel_count(
            rgb_values,
            self.rgb_intrinsics,
            self._latest_floor[2],
            config,
        )
