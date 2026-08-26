"""ROS adapter for independently stamped geometry and RGB cable evidence."""

from __future__ import annotations

import math
import struct
import sys
from array import array

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

from .cable import DiagnosticPinkConfig, TrainingFreeCableConfig
from .depth_evidence import DepthEvidenceConfig
from .geometric_pipeline import CandidateReport, GeometricHazardPipeline
from .geometry import (
    CameraIntrinsics,
    GroundEstimate,
    GroundEstimatorConfig,
    StrongGeometryConfig,
)
from .health import (
    CameraInfoObservation,
    HealthSnapshot,
    HealthState,
    ImageObservation,
    OdomObservation,
    Stream,
    TransformBatchObservation,
    geometric_projection_support_reason,
    rgb_projection_support_reason,
)
from .node import InputHealthNode, _stamp_ns
from .temporal import (
    ConfirmedHazard,
    EvidenceMask,
    EvidenceSource,
    HazardTrackerConfig,
    Pose3,
)


class GeometricHazardNode(InputHealthNode):
    """Carry supported low-profile hazards through odom to PointCloud2."""

    def __init__(self) -> None:
        self._pipeline = GeometricHazardPipeline()
        self._pending_depth: tuple[int, tuple[int, ...]] | None = None
        self._pending_rgb: tuple[int, bytes] | None = None
        self._last_ground: GroundEstimate | None = None
        self._last_nominal_ground_angle_error_degrees: float | None = None
        self._last_candidate_count = 0
        self._confirmed_observation_count = 0
        self._latest_confirmation_spread_m: float | None = None
        self._cloud_publish_count = 0
        self._candidate_publish_count = 0
        self._pending_depth_drop_count = 0
        self._processed_depth_count = 0
        self._pending_rgb_drop_count = 0
        self._processed_rgb_count = 0
        self._latest_processed_rgb_stamp_ns = 0
        self._last_cable_candidate_count = 0
        self._rgb_cable_confirmation_count = 0
        self._diagnostic_pink_pixel_count: int | None = None
        self._latest_processed_depth_stamp_ns = 0
        self._geometric_degradation_reason = ""
        self._last_published_retained_signature: tuple[object, ...] = ()
        self._had_operational_hazard_output = False
        self._active_retained_count = 0
        super().__init__()

        self._diagnostic_pink_enabled = bool(
            self.declare_parameter("diagnostic_pink_comparison_enabled", False).value
        )
        self._diagnostic_pink_config = DiagnosticPinkConfig(
            minimum_red=int(
                self.declare_parameter("diagnostic_pink_minimum_red", 90).value
            ),
            minimum_red_over_green=int(
                self.declare_parameter(
                    "diagnostic_pink_minimum_red_over_green", 10
                ).value
            ),
            minimum_blue_over_green=int(
                self.declare_parameter(
                    "diagnostic_pink_minimum_blue_over_green", 5
                ).value
            ),
            maximum_red_blue_difference=int(
                self.declare_parameter(
                    "diagnostic_pink_maximum_red_blue_difference", 60
                ).value
            ),
        )

        ground_config = GroundEstimatorConfig(
            sample_stride_px=int(
                self.declare_parameter("ground_sample_stride_px", 6).value
            ),
            ransac_iterations=int(
                self.declare_parameter("ground_ransac_iterations", 160).value
            ),
            ransac_score_max_samples=int(
                self.declare_parameter("ground_ransac_score_max_samples", 1200).value
            ),
            inlier_threshold_m=float(
                self.declare_parameter("ground_inlier_threshold_m", 0.008).value
            ),
            minimum_support=int(
                self.declare_parameter("ground_minimum_support", 500).value
            ),
            minimum_inlier_ratio=float(
                self.declare_parameter("ground_minimum_inlier_ratio", 0.70).value
            ),
            maximum_p90_residual_m=float(
                self.declare_parameter("ground_maximum_p90_residual_m", 0.008).value
            ),
            minimum_spatial_coverage=float(
                self.declare_parameter("ground_minimum_spatial_coverage", 0.35).value
            ),
            minimum_temporal_consistency=float(
                self.declare_parameter(
                    "ground_minimum_temporal_consistency", 0.20
                ).value
            ),
            temporal_smoothing_factor=float(
                self.declare_parameter("ground_temporal_smoothing_factor", 0.35).value
            ),
        )
        geometry_config = StrongGeometryConfig(
            sample_stride_px=int(
                self.declare_parameter("geometry_sample_stride_px", 2).value
            ),
            strong_height_m=float(
                self.declare_parameter("strong_height_m", 0.015).value
            ),
            maximum_height_m=float(
                self.declare_parameter("maximum_hazard_height_m", 0.15).value
            ),
            minimum_support_points=int(
                self.declare_parameter("strong_minimum_support_points", 18).value
            ),
            cluster_cell_m=float(
                self.declare_parameter("geometry_cluster_cell_m", 0.04).value
            ),
            minimum_spatial_span_m=float(
                self.declare_parameter("strong_minimum_spatial_span_m", 0.04).value
            ),
        )
        depth_evidence_config = DepthEvidenceConfig(
            sample_stride_px=int(
                self.declare_parameter("depth_evidence_sample_stride_px", 1).value
            ),
            minimum_weak_height_m=float(
                self.declare_parameter("weak_height_m", 0.006).value
            ),
            ground_noise_multiplier=float(
                self.declare_parameter("weak_ground_noise_multiplier", 3.0).value
            ),
            strong_height_m=geometry_config.strong_height_m,
            minimum_weak_support_points=int(
                self.declare_parameter("weak_minimum_support_points", 8).value
            ),
            cluster_cell_m=geometry_config.cluster_cell_m,
            minimum_physical_span_m=float(
                self.declare_parameter("weak_minimum_spatial_span_m", 0.04).value
            ),
            minimum_invalid_pixels=int(
                self.declare_parameter("invalid_depth_minimum_pixels", 12).value
            ),
            minimum_invalid_span_px=float(
                self.declare_parameter("invalid_depth_minimum_span_px", 12.0).value
            ),
            maximum_invalid_width_px=float(
                self.declare_parameter("invalid_depth_maximum_width_px", 8.0).value
            ),
        )
        cable_config = TrainingFreeCableConfig(
            local_contrast_threshold=float(
                self.declare_parameter("cable_local_contrast_threshold", 24.0).value
            ),
            maximum_half_width_px=int(
                self.declare_parameter("cable_maximum_half_width_px", 3).value
            ),
            minimum_component_pixels=int(
                self.declare_parameter("cable_minimum_component_pixels", 16).value
            ),
            minimum_length_px=float(
                self.declare_parameter("cable_minimum_length_px", 16.0).value
            ),
            minimum_apparent_width_px=float(
                self.declare_parameter("cable_minimum_apparent_width_px", 1.5).value
            ),
            maximum_apparent_width_px=float(
                self.declare_parameter("cable_maximum_apparent_width_px", 6.0).value
            ),
            minimum_width_consistency=float(
                self.declare_parameter("cable_minimum_width_consistency", 0.55).value
            ),
            minimum_curve_continuity=float(
                self.declare_parameter("cable_minimum_curve_continuity", 0.80).value
            ),
            minimum_physical_span_m=float(
                self.declare_parameter("cable_minimum_physical_span_m", 0.06).value
            ),
            maximum_ground_age_ns=int(
                float(
                    self.declare_parameter("cable_maximum_ground_age_ms", 500.0).value
                )
                * 1_000_000
            ),
        )
        tracker_config = HazardTrackerConfig(
            association_radius_m=float(
                self.declare_parameter("association_radius_m", 0.08).value
            ),
            confirmation_window_ns=int(
                float(self.declare_parameter("confirmation_window_ms", 350.0).value)
                * 1_000_000
            ),
            candidate_retention_ns=int(
                float(self.declare_parameter("candidate_retention_ms", 500.0).value)
                * 1_000_000
            ),
            confirmed_retention_ns=int(
                float(self.declare_parameter("confirmed_retention_ms", 2000.0).value)
                * 1_000_000
            ),
            minimum_rgb_confirmation_confidence=float(
                self.declare_parameter(
                    "minimum_rgb_confirmation_confidence", 0.75
                ).value
            ),
        )
        self._pipeline = GeometricHazardPipeline(
            ground_config=ground_config,
            geometry_config=geometry_config,
            depth_evidence_config=depth_evidence_config,
            cable_config=cable_config,
            tracker_config=tracker_config,
        )
        cloud_topic = str(
            self.declare_parameter(
                "hazard_cloud_topic",
                "/low_profile_hazard_perception/confirmed_hazards",
            ).value
        )
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._cloud_publisher = self.create_publisher(
            PointCloud2, cloud_topic, cloud_qos
        )
        candidate_topic = str(
            self.declare_parameter(
                "candidate_diagnostics_topic",
                "/low_profile_hazard_perception/candidate_diagnostics",
            ).value
        )
        self._candidate_publisher = self.create_publisher(
            DiagnosticArray,
            candidate_topic,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        retention_check_period_ms = float(
            self.declare_parameter("retention_check_period_ms", 100.0).value
        )
        if retention_check_period_ms <= 0.0:
            raise ValueError("retention_check_period_ms must be positive")
        self._retention_timer = self.create_timer(
            retention_check_period_ms / 1000.0,
            self._publish_retained_at_sensor_now,
        )

    def _after_camera_info(
        self, stream: Stream, observation: CameraInfoObservation
    ) -> None:
        if stream not in (
            Stream.COLOR_CAMERA_INFO,
            Stream.DEPTH_CAMERA_INFO,
        ):
            return
        if (
            observation.frame_id != self._camera_frame
            or observation.width <= 0
            or observation.height <= 0
            or not all(
                math.isfinite(value)
                for value in (
                    observation.fx,
                    observation.fy,
                    observation.cx,
                    observation.cy,
                )
            )
            or observation.fx <= 0.0
            or observation.fy <= 0.0
        ):
            self._block_new_confirmation("camera_info:invalid")
            return
        intrinsics = CameraIntrinsics(
            width=observation.width,
            height=observation.height,
            fx=observation.fx,
            fy=observation.fy,
            cx=observation.cx,
            cy=observation.cy,
        )
        if stream is Stream.DEPTH_CAMERA_INFO:
            self._pipeline.set_intrinsics(intrinsics)
            self._try_process_pending_depth()
        else:
            self._pipeline.set_rgb_intrinsics(intrinsics)
            self._try_process_pending_rgb()

    def _before_odom(self, message: Odometry) -> None:
        stamp_ns = _stamp_ns(message.header.stamp)
        if (
            stamp_ns <= 0
            or message.header.frame_id != "odom"
            or message.child_frame_id != self._base_frame
        ):
            self._block_new_confirmation("odom:invalid")
            return
        pose = message.pose.pose
        try:
            reason = self._pipeline.add_odom(
                stamp_ns,
                Pose3(
                    translation=(
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                    ),
                    rotation=(
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ),
                ),
            )
        except ValueError:
            self._block_new_confirmation("odom:invalid")
            return
        if reason:
            self._geometric_degradation_reason = reason

    def _after_odom(self, observation: OdomObservation) -> None:
        del observation
        self._try_process_pending_depth()
        self._try_process_pending_rgb()

    def _after_tf(
        self,
        stream: Stream,
        is_static: bool,
        message: TFMessage,
        observation: TransformBatchObservation,
    ) -> None:
        del stream, is_static, message
        if observation.input_error or not observation.required_chain_available:
            self._block_new_confirmation(
                "tf:invalid" if observation.input_error else "tf:chain_unavailable"
            )
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame, Time()
            ).transform
            self._pipeline.set_base_from_camera(
                Pose3(
                    translation=(
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ),
                    rotation=(
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                        transform.rotation.w,
                    ),
                )
            )
        except Exception as error:
            self.get_logger().warning(
                f"camera mounting transform is not usable: {error}"
            )
            self._block_new_confirmation("tf:unusable")
            return
        self._try_process_pending_depth()
        self._try_process_pending_rgb()

    def _after_image(
        self,
        stream: Stream,
        message: Image,
        observation: ImageObservation,
    ) -> None:
        if stream is Stream.COLOR_IMAGE:
            if (
                observation.sensor_stamp_ns <= 0
                or observation.frame_id != self._camera_frame
                or observation.encoding != "rgb8"
                or observation.step < observation.width * 3
                or observation.data_size < observation.step * observation.height
            ):
                self._block_new_confirmation("color:invalid")
                return
            try:
                values = _rgb_values(message)
            except ValueError as error:
                self.get_logger().warning(f"RGB cable path rejected frame: {error}")
                return
            if self._pending_rgb is not None:
                self._pending_rgb_drop_count += 1
            self._pending_rgb = (observation.sensor_stamp_ns, values)
            self._try_process_pending_rgb()
            return
        if stream is not Stream.DEPTH_IMAGE:
            return
        if (
            observation.sensor_stamp_ns <= 0
            or observation.frame_id != self._camera_frame
            or observation.encoding != "16UC1"
            or observation.step < observation.width * 2
            or observation.step % 2 != 0
            or observation.data_size < observation.step * observation.height
        ):
            self._block_new_confirmation("depth:invalid")
            return
        try:
            values = _depth_values(message)
        except ValueError as error:
            self.get_logger().warning(f"depth geometry rejected frame: {error}")
            return
        if self._pending_depth is not None:
            self._pending_depth_drop_count += 1
        self._pending_depth = (observation.sensor_stamp_ns, values)
        self._try_process_pending_depth()

    def _try_process_pending_depth(self) -> None:
        if self._pending_depth is None:
            return
        stamp_ns, values = self._pending_depth
        snapshot = self._snapshot()
        projection_reason = geometric_projection_support_reason(snapshot)
        if projection_reason:
            self._block_new_confirmation(projection_reason)
            return
        if self._pipeline.alignment_at(stamp_ns) is None:
            self._geometric_degradation_reason = (
                self._pipeline.alignment_reason or "odom:missing"
            )
            self._publish_health()
            return
        self._pending_depth = None
        result = self._pipeline.process_depth(stamp_ns, values)
        if result is None:
            self._geometric_degradation_reason = (
                self._pipeline.alignment_reason or "odom:missing"
            )
            self._publish_health()
            return
        self._geometric_degradation_reason = result.degradation_reason
        self._processed_depth_count += 1
        self._latest_processed_depth_stamp_ns = stamp_ns
        self._last_ground = result.ground
        self._last_nominal_ground_angle_error_degrees = (
            result.nominal_ground_angle_error_degrees
        )
        self._last_candidate_count = len(result.candidates)
        if result.confirmed:
            self._confirmed_observation_count += len(result.confirmed)
            self._latest_confirmation_spread_m = max(
                confirmed.spatial_spread_m for confirmed in result.confirmed
            )
        self._publish_candidate_reports(result.candidate_reports)
        self._publish_retained(result.retained, sensor_now_ns=stamp_ns)
        self._publish_health()
        self._try_process_pending_rgb()

    def _try_process_pending_rgb(self) -> None:
        if self._pending_rgb is None:
            return
        stamp_ns, values = self._pending_rgb
        projection_reason = rgb_projection_support_reason(self._snapshot())
        if projection_reason:
            self._block_new_confirmation(projection_reason)
            return
        if self._pipeline.alignment_at(stamp_ns) is None:
            self._geometric_degradation_reason = (
                self._pipeline.alignment_reason or "odom:missing"
            )
            self._publish_health()
            return
        result = self._pipeline.process_rgb(stamp_ns, values)
        if result is None:
            self._geometric_degradation_reason = "ground:unavailable"
            self._publish_health()
            return
        if self._diagnostic_pink_enabled:
            self._diagnostic_pink_pixel_count = self._pipeline.diagnostic_pink_count(
                values,
                self._diagnostic_pink_config,
            )
        self._pending_rgb = None
        self._geometric_degradation_reason = result.degradation_reason
        self._processed_rgb_count += 1
        self._latest_processed_rgb_stamp_ns = stamp_ns
        self._last_cable_candidate_count = len(result.candidates)
        rgb_confirmed = [
            confirmed
            for confirmed in result.confirmed
            if EvidenceSource.RGB_CABLE in confirmed.evidence
        ]
        if rgb_confirmed:
            self._rgb_cable_confirmation_count += len(rgb_confirmed)
            self._confirmed_observation_count += len(rgb_confirmed)
            self._latest_confirmation_spread_m = max(
                confirmed.spatial_spread_m for confirmed in rgb_confirmed
            )
        self._publish_candidate_reports(result.candidate_reports)
        self._publish_retained(result.retained, sensor_now_ns=stamp_ns)
        self._publish_health()

    def _block_new_confirmation(self, reason: str) -> None:
        self._pipeline.suspend_confirmed_expiry()
        self._geometric_degradation_reason = reason
        self._publish_health()

    def _publish_retained_at_sensor_now(self) -> None:
        sensor_now_ns = self.get_clock().now().nanoseconds
        snapshot = self._snapshot()
        if (
            snapshot.state is not HealthState.HEALTHY
            or self._geometric_degradation_reason
            or (self._last_ground is not None and not self._last_ground.accepted)
        ):
            self._pipeline.suspend_confirmed_expiry()
        self._publish_retained(
            self._pipeline.retained_at(sensor_now_ns),
            sensor_now_ns=sensor_now_ns,
        )

    def _publish_retained(
        self,
        retained: tuple[ConfirmedHazard, ...],
        *,
        sensor_now_ns: int,
    ) -> None:
        signature: tuple[object, ...] = tuple(
            (
                hazard.sensor_stamp_ns,
                hazard.points_odom,
                hazard.spatial_spread_m,
                hazard.evidence,
                hazard.confidence,
            )
            for hazard in retained
        )
        if signature == self._last_published_retained_signature:
            self._active_retained_count = len(retained)
            return
        if not retained and not self._had_operational_hazard_output:
            self._last_published_retained_signature = signature
            self._active_retained_count = 0
            return
        self._cloud_publisher.publish(
            _point_cloud(retained, sensor_now_ns=sensor_now_ns)
        )
        self._cloud_publish_count += 1
        self._had_operational_hazard_output = bool(retained)
        self._active_retained_count = len(retained)
        self._last_published_retained_signature = signature

    def _publish_candidate_reports(self, reports: tuple[CandidateReport, ...]) -> None:
        if not reports:
            return
        message = DiagnosticArray()
        latest_stamp_ns = max(report.sensor_stamp_ns for report in reports)
        message.header.stamp = Time(nanoseconds=latest_stamp_ns).to_msg()
        message.status = [
            _candidate_diagnostic(report, index) for index, report in enumerate(reports)
        ]
        self._candidate_publisher.publish(message)
        self._candidate_publish_count += 1

    def _operational_hazard_output_enabled(self) -> bool:
        return True

    def _effective_health_state(self, snapshot: HealthSnapshot) -> HealthState:
        if snapshot.state is HealthState.INVALID:
            return HealthState.INVALID
        if self._geometric_degradation_reason or (
            self._last_ground is not None and not self._last_ground.accepted
        ):
            return HealthState.DEGRADED
        return snapshot.state

    def _additional_health_reasons(self) -> tuple[str, ...]:
        if self._geometric_degradation_reason:
            return (self._geometric_degradation_reason,)
        if self._last_ground is None or self._last_ground.accepted:
            return ()
        return (f"ground:{self._last_ground.reason}",)

    def _additional_health_values(self) -> dict[str, object]:
        if self._last_ground is None:
            ground: dict[str, object] = {
                "ground.state": "UNAVAILABLE",
                "ground.reason": "no processed depth observation",
            }
        else:
            estimate = self._last_ground
            metrics = estimate.metrics
            ground = {
                "ground.state": "VALID" if estimate.accepted else "REJECTED",
                "ground.reason": estimate.reason,
                "ground.support_count": metrics.support_count,
                "ground.sampled_valid_count": metrics.sampled_valid_count,
                "ground.inlier_ratio": metrics.inlier_ratio,
                "ground.median_residual_m": metrics.median_residual_m,
                "ground.p90_residual_m": metrics.p90_residual_m,
                "ground.spatial_coverage": metrics.spatial_coverage,
                "ground.temporal_consistency": (metrics.temporal_consistency),
                "ground.camera_height_m": estimate.model.camera_height_m,
                "ground.nominal_height_error_m": (metrics.nominal_height_error_m),
                "ground.nominal_angle_error_degrees": (
                    self._last_nominal_ground_angle_error_degrees
                ),
            }
        ground.update(
            {
                "geometry.latest_candidate_count": self._last_candidate_count,
                "geometry.confirmed_observation_count": (
                    self._confirmed_observation_count
                ),
                "geometry.latest_confirmation_spread_m": (
                    self._latest_confirmation_spread_m
                ),
                "geometry.cloud_publish_count": self._cloud_publish_count,
                "geometry.candidate_diagnostics_publish_count": (
                    self._candidate_publish_count
                ),
                "geometry.pending_depth_drops": (self._pending_depth_drop_count),
                "geometry.processed_depth_count": self._processed_depth_count,
                "geometry.latest_processed_depth_stamp_ns": (
                    self._latest_processed_depth_stamp_ns
                ),
                "cable.provider": "training_free_thin_line",
                "cable.latest_candidate_count": (self._last_cable_candidate_count),
                "cable.confirmed_observation_count": (
                    self._rgb_cable_confirmation_count
                ),
                "cable.confirmation_observations": 2,
                "cable.minimum_confirmation_confidence": (
                    self._pipeline.tracker.config.minimum_rgb_confirmation_confidence
                ),
                "cable.pending_rgb_drops": self._pending_rgb_drop_count,
                "cable.processed_rgb_count": self._processed_rgb_count,
                "cable.latest_processed_rgb_stamp_ns": (
                    self._latest_processed_rgb_stamp_ns
                ),
                "cable.rgb_depth_synchronizer": "disabled",
                "cable.projection": "ray_observed_ground",
                "cable.diagnostic_pink_comparison_enabled": (
                    self._diagnostic_pink_enabled
                ),
                "cable.diagnostic_pink_pixel_count": (
                    self._diagnostic_pink_pixel_count
                ),
                "cable.diagnostic_pink_operational": False,
                "geometry.active_retained_hazard_count": (self._active_retained_count),
                "geometry.degradation_reason": (self._geometric_degradation_reason),
                "geometry.alignment_frame": "odom",
                "geometry.confirmation_observations": 2,
                "geometry.candidate_retention_ms": (
                    self._pipeline.tracker.config.candidate_retention_ns / 1_000_000
                ),
                "geometry.confirmed_retention_ms": (
                    self._pipeline.tracker.config.confirmed_retention_ns / 1_000_000
                ),
                "geometry.output_durability": "transient_local",
                "odom.maximum_interpolation_gap_ms": (
                    self._pipeline.odom.maximum_interpolation_gap_ns / 1_000_000
                ),
                "odom.maximum_translation_jump_m": (
                    self._pipeline.odom.maximum_translation_jump_m
                ),
                "odom.maximum_rotation_jump_degrees": (
                    self._pipeline.odom.maximum_rotation_jump_degrees
                ),
            }
        )
        return ground


def _candidate_diagnostic(report: CandidateReport, index: int) -> DiagnosticStatus:
    status = DiagnosticStatus()
    confirmed = report.decision_reason.value.startswith("CONFIRMED_")
    status.level = DiagnosticStatus.OK if confirmed else DiagnosticStatus.WARN
    status.name = f"low_profile_hazard_perception/candidate/{index}"
    status.message = report.decision_reason.value
    status.hardware_id = "rgbd_evidence_fusion"
    metrics = report.ground_quality
    values = {
        "sensor_stamp_ns": report.sensor_stamp_ns,
        "centroid_odom_x": report.centroid_odom[0],
        "centroid_odom_y": report.centroid_odom[1],
        "centroid_odom_z": report.centroid_odom[2],
        "evidence_sources": ",".join(
            source.value for source in report.evidence_sources
        ),
        "ground.accepted": report.ground_accepted,
        "ground.reason": report.ground_reason,
        "ground.support_count": metrics.support_count,
        "ground.inlier_ratio": metrics.inlier_ratio,
        "ground.p90_residual_m": metrics.p90_residual_m,
        "ground.spatial_coverage": metrics.spatial_coverage,
        "ground.temporal_consistency": metrics.temporal_consistency,
        "confidence": report.confidence,
        "decision_reason": report.decision_reason.value,
        "operational_confirmed": confirmed,
    }
    status.values = [
        KeyValue(key=key, value=_diagnostic_value(value))
        for key, value in values.items()
    ]
    return status


def _diagnostic_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _depth_values(message: Image) -> tuple[int, ...]:
    raw = array("H")
    raw.frombytes(bytes(message.data))
    message_is_big_endian = bool(message.is_bigendian)
    if message_is_big_endian == (sys.byteorder == "little"):
        raw.byteswap()
    row_words = message.step // 2
    required_words = row_words * message.height
    if len(raw) < required_words:
        raise ValueError("depth payload is shorter than its stride")
    if row_words == message.width:
        return tuple(raw[: message.width * message.height])
    return tuple(
        raw[row * row_words + column]
        for row in range(message.height)
        for column in range(message.width)
    )


def _rgb_values(message: Image) -> bytes:
    row_bytes = message.width * 3
    required_bytes = message.step * message.height
    if len(message.data) < required_bytes:
        raise ValueError("RGB payload is shorter than its stride")
    if message.step == row_bytes:
        return bytes(message.data[: row_bytes * message.height])
    return b"".join(
        bytes(message.data[row * message.step : row * message.step + row_bytes])
        for row in range(message.height)
    )


def _point_cloud(
    retained: tuple[ConfirmedHazard, ...],
    *,
    sensor_now_ns: int,
) -> PointCloud2:
    cloud = PointCloud2()
    observation_stamp_ns = min(
        (hazard.sensor_stamp_ns for hazard in retained),
        default=sensor_now_ns,
    )
    cloud.header.stamp = Time(nanoseconds=observation_stamp_ns).to_msg()
    cloud.header.frame_id = "odom"
    points = tuple(
        (
            point,
            hazard.spatial_spread_m,
            divmod(hazard.sensor_stamp_ns, 1_000_000_000),
            int(EvidenceMask.from_sources(hazard.evidence)),
        )
        for hazard in retained
        for point in hazard.points_odom
    )
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="confirmation_spread",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="observation_stamp_sec",
            offset=16,
            datatype=PointField.INT32,
            count=1,
        ),
        PointField(
            name="observation_stamp_nanosec",
            offset=20,
            datatype=PointField.UINT32,
            count=1,
        ),
        PointField(
            name="evidence_mask",
            offset=24,
            datatype=PointField.UINT8,
            count=1,
        ),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 28
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = b"".join(
        struct.pack(
            "<ffffiIB3x",
            *point,
            confirmation_spread_m,
            stamp_parts[0],
            stamp_parts[1],
            evidence_mask,
        )
        for point, confirmation_spread_m, stamp_parts, evidence_mask in points
    )
    cloud.is_dense = True
    return cloud


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GeometricHazardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
