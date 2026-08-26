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
from .depth_evidence import DepthEvidenceConfig, DepthEvidenceDetector
from .geometry import (
    CameraIntrinsics,
    GeometricCandidate,
    GroundEstimate,
    GroundQualityMetrics,
    GroundEstimator,
    GroundEstimatorConfig,
    StrongGeometryConfig,
    StrongGeometryDetector,
)
from .temporal import (
    CandidateDecisionReason,
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
    candidate_reports: tuple["CandidateReport", ...]
    nominal_ground_angle_error_degrees: float | None
    degradation_reason: str = ""


@dataclass(frozen=True)
class RgbCablePipelineResult:
    sensor_stamp_ns: int
    ground_sensor_stamp_ns: int
    candidates: tuple[CableCandidate, ...]
    confirmed: tuple[ConfirmedHazard, ...]
    retained: tuple[ConfirmedHazard, ...]
    candidate_reports: tuple["CandidateReport", ...]
    degradation_reason: str = ""


@dataclass(frozen=True)
class ObservedFloorSnapshot:
    sensor_stamp_ns: int
    estimate: GroundEstimate
    region: ObservedFloorRegion


@dataclass(frozen=True)
class CandidateReport:
    sensor_stamp_ns: int
    centroid_odom: tuple[float, float, float]
    evidence_sources: tuple[EvidenceSource, ...]
    ground_accepted: bool
    ground_reason: str
    ground_quality: GroundQualityMetrics
    confidence: float
    decision_reason: CandidateDecisionReason


class GeometricHazardPipeline:
    """Own the explicitly asynchronous geometry/alignment/confirmation path."""

    def __init__(
        self,
        *,
        ground_config: GroundEstimatorConfig | None = None,
        geometry_config: StrongGeometryConfig | None = None,
        depth_evidence_config: DepthEvidenceConfig | None = None,
        cable_config: TrainingFreeCableConfig | None = None,
        tracker_config: HazardTrackerConfig | None = None,
    ) -> None:
        self.ground_estimator = GroundEstimator(ground_config)
        self.geometry_detector = StrongGeometryDetector(geometry_config)
        self.depth_evidence_detector = DepthEvidenceDetector(depth_evidence_config)
        self.cable_detector = TrainingFreeCableDetector(cable_config)
        self.odom = OdomPoseCache()
        self.tracker = HazardTracker(tracker_config)
        self.intrinsics: CameraIntrinsics | None = None
        self.rgb_intrinsics: CameraIntrinsics | None = None
        self.base_from_camera: Pose3 | None = None
        self.nominal_camera_height_m = 0.0
        self._last_aligned_observation_stamp_ns: int | None = None
        self._last_aligned_rgb_stamp_ns: int | None = None
        self._latest_floor: ObservedFloorSnapshot | None = None
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

    def retained_at(self, sensor_now_ns: int) -> tuple[ConfirmedHazard, ...]:
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
        nominal_angle_error = self.base_from_camera.normal_error_to_parent_up_degrees(
            ground.model.normal
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
                candidate_reports=(),
                nominal_ground_angle_error_degrees=nominal_angle_error,
                degradation_reason=f"ground:{ground.reason}",
            )
        observed_base_from_camera = self.base_from_camera.with_observed_ground(
            ground.model.normal, ground.model.camera_height_m
        )
        floor_region = ObservedFloorRegion.from_depth(
            depth_values,
            self.intrinsics,
            ground.model,
            depth_unit_m=depth_unit_m,
        )
        self._latest_floor = ObservedFloorSnapshot(
            sensor_stamp_ns=sensor_stamp_ns,
            estimate=ground,
            region=floor_region,
        )
        odom_from_camera = odom_from_base.compose(observed_base_from_camera)
        candidates = self.geometry_detector.detect(
            depth_values,
            self.intrinsics,
            ground.model,
            depth_unit_m=depth_unit_m,
        )
        depth_evidence = self.depth_evidence_detector.detect(
            depth_values,
            self.intrinsics,
            ground.model,
            floor_region,
            depth_unit_m=depth_unit_m,
            ground_noise_m=ground.metrics.p90_residual_m / 1.645,
        )
        confirmed, candidate_reports, degradation_reason = self._track_observations(
            tuple(
                HazardObservation(
                    sensor_stamp_ns=sensor_stamp_ns,
                    points_odom=tuple(
                        _stable_point(odom_from_camera.transform_point(point))
                        for point in candidate.points
                    ),
                    evidence=EvidenceSource.STRONG_GEOMETRY,
                    confidence=candidate.confidence,
                )
                for candidate in candidates
            )
            + tuple(
                HazardObservation(
                    sensor_stamp_ns=sensor_stamp_ns,
                    points_odom=tuple(
                        _stable_point(odom_from_camera.transform_point(point))
                        for point in candidate.points_camera
                    ),
                    evidence=candidate.evidence,
                    confidence=candidate.confidence,
                )
                for candidate in depth_evidence
            ),
            ground=ground,
            degradation_reason=degradation_reason,
        )
        return GeometricPipelineResult(
            sensor_stamp_ns=sensor_stamp_ns,
            ground=ground,
            candidates=candidates,
            confirmed=tuple(confirmed),
            retained=self.tracker.retained_at(
                sensor_stamp_ns,
                allow_confirmed_expiry=(not self._confirmed_expiry_suspended),
            ),
            candidate_reports=candidate_reports,
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
        floor = self._latest_floor
        if (
            abs(sensor_stamp_ns - floor.sensor_stamp_ns)
            > self.cable_detector.config.maximum_ground_age_ns
        ):
            self._suspend_confirmation()
            return RgbCablePipelineResult(
                sensor_stamp_ns=sensor_stamp_ns,
                ground_sensor_stamp_ns=floor.sensor_stamp_ns,
                candidates=(),
                confirmed=(),
                retained=self.tracker.retained_at(
                    sensor_stamp_ns,
                    allow_confirmed_expiry=False,
                ),
                candidate_reports=(),
                degradation_reason="ground:stale",
            )
        candidates = self.cable_detector.detect(
            rgb_values,
            self.rgb_intrinsics,
            floor.estimate.model,
            floor.region,
        )
        observed_base_from_camera = self.base_from_camera.with_observed_ground(
            floor.estimate.model.normal,
            floor.estimate.model.camera_height_m,
        )
        odom_from_camera = alignment.pose.compose(observed_base_from_camera)
        confirmed, candidate_reports, degradation_reason = self._track_observations(
            tuple(
                HazardObservation(
                    sensor_stamp_ns=sensor_stamp_ns,
                    points_odom=tuple(
                        _stable_point(odom_from_camera.transform_point(point))
                        for point in candidate.points_camera
                    ),
                    evidence=EvidenceSource.RGB_CABLE,
                    confidence=candidate.confidence,
                )
                for candidate in candidates
            ),
            ground=floor.estimate,
            degradation_reason=degradation_reason,
        )
        return RgbCablePipelineResult(
            sensor_stamp_ns=sensor_stamp_ns,
            ground_sensor_stamp_ns=floor.sensor_stamp_ns,
            candidates=candidates,
            confirmed=tuple(confirmed),
            retained=self.tracker.retained_at(
                sensor_stamp_ns,
                allow_confirmed_expiry=(not self._confirmed_expiry_suspended),
            ),
            candidate_reports=candidate_reports,
            degradation_reason=degradation_reason,
        )

    def _track_observations(
        self,
        observations: tuple[HazardObservation, ...],
        *,
        ground: GroundEstimate,
        degradation_reason: str,
    ) -> tuple[tuple[ConfirmedHazard, ...], tuple[CandidateReport, ...], str,]:
        """Apply one shared confirmation/recovery lifecycle to all evidence."""
        reconfirmation_required = self._confirmed_expiry_suspended
        confirmed: list[ConfirmedHazard] = []
        reports: list[CandidateReport] = []
        for observation in observations:
            decision = self.tracker.observe_with_decision(
                observation,
                allow_confirmed_expiry=(not self._confirmed_expiry_suspended),
                require_reconfirmation_for_confirmed=(reconfirmation_required),
            )
            confirmed.extend(decision.confirmed)
            reports.append(
                CandidateReport(
                    sensor_stamp_ns=observation.sensor_stamp_ns,
                    centroid_odom=observation.centroid,
                    evidence_sources=decision.evidence,
                    ground_accepted=ground.accepted,
                    ground_reason=ground.reason,
                    ground_quality=ground.metrics,
                    confidence=decision.confidence,
                    decision_reason=decision.decision_reason,
                )
            )
        if confirmed and not degradation_reason:
            self._confirmed_expiry_suspended = False
        elif reconfirmation_required and not degradation_reason:
            degradation_reason = "recovery:reconfirmation_required"
        return tuple(confirmed), tuple(reports), degradation_reason

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
            self._latest_floor.region,
            config,
        )


def _stable_point(
    point: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Keep odom evidence deterministic below meaningful spatial precision."""
    return tuple(round(value, 4) for value in point)
