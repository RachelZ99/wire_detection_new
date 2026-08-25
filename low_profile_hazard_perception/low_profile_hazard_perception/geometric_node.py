"""ROS adapter for strong depth geometry at independent capture times."""

from __future__ import annotations

import math
import struct
import sys
from array import array

import rclpy
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

from .geometric_pipeline import GeometricHazardPipeline
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
    Stream,
    TransformBatchObservation,
)
from .node import InputHealthNode, _stamp_ns
from .temporal import HazardTrackerConfig, Pose3


class GeometricHazardNode(InputHealthNode):
    """Carry supported protrusions through odom confirmation to PointCloud2."""

    def __init__(self) -> None:
        self._pipeline = GeometricHazardPipeline()
        self._pending_depth: tuple[int, tuple[int, ...]] | None = None
        self._last_ground: GroundEstimate | None = None
        self._last_candidate_count = 0
        self._confirmed_observation_count = 0
        self._latest_confirmation_spread_m: float | None = None
        self._cloud_publish_count = 0
        self._pending_depth_drop_count = 0
        super().__init__()

        ground_config = GroundEstimatorConfig(
            sample_stride_px=int(
                self.declare_parameter("ground_sample_stride_px", 6).value
            ),
            ransac_iterations=int(
                self.declare_parameter("ground_ransac_iterations", 160).value
            ),
            ransac_score_max_samples=int(
                self.declare_parameter(
                    "ground_ransac_score_max_samples", 1200
                ).value
            ),
            inlier_threshold_m=float(
                self.declare_parameter(
                    "ground_inlier_threshold_m", 0.008
                ).value
            ),
            minimum_support=int(
                self.declare_parameter("ground_minimum_support", 500).value
            ),
            minimum_inlier_ratio=float(
                self.declare_parameter(
                    "ground_minimum_inlier_ratio", 0.70
                ).value
            ),
            maximum_p90_residual_m=float(
                self.declare_parameter(
                    "ground_maximum_p90_residual_m", 0.008
                ).value
            ),
            minimum_spatial_coverage=float(
                self.declare_parameter(
                    "ground_minimum_spatial_coverage", 0.35
                ).value
            ),
            minimum_temporal_consistency=float(
                self.declare_parameter(
                    "ground_minimum_temporal_consistency", 0.20
                ).value
            ),
            temporal_smoothing_factor=float(
                self.declare_parameter(
                    "ground_temporal_smoothing_factor", 0.35
                ).value
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
                self.declare_parameter(
                    "strong_minimum_support_points", 18
                ).value
            ),
            cluster_cell_m=float(
                self.declare_parameter("geometry_cluster_cell_m", 0.04).value
            ),
            minimum_spatial_span_m=float(
                self.declare_parameter(
                    "strong_minimum_spatial_span_m", 0.04
                ).value
            ),
        )
        tracker_config = HazardTrackerConfig(
            association_radius_m=float(
                self.declare_parameter("association_radius_m", 0.08).value
            ),
            confirmation_window_ns=int(
                float(
                    self.declare_parameter(
                        "confirmation_window_ms", 350.0
                    ).value
                )
                * 1_000_000
            ),
        )
        self._pipeline = GeometricHazardPipeline(
            ground_config=ground_config,
            geometry_config=geometry_config,
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
            durability=DurabilityPolicy.VOLATILE,
        )
        self._cloud_publisher = self.create_publisher(
            PointCloud2, cloud_topic, cloud_qos
        )

    def _after_camera_info(
        self, stream: Stream, observation: CameraInfoObservation
    ) -> None:
        if stream is not Stream.DEPTH_CAMERA_INFO:
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
            return
        self._pipeline.set_intrinsics(
            CameraIntrinsics(
                width=observation.width,
                height=observation.height,
                fx=observation.fx,
                fy=observation.fy,
                cx=observation.cx,
                cy=observation.cy,
            )
        )
        self._try_process_pending_depth()

    def _before_odom(self, message: Odometry) -> None:
        stamp_ns = _stamp_ns(message.header.stamp)
        if (
            stamp_ns <= 0
            or message.header.frame_id != "odom"
            or message.child_frame_id != self._base_frame
        ):
            return
        pose = message.pose.pose
        try:
            self._pipeline.add_odom(
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
            return
        self._try_process_pending_depth()

    def _after_tf(
        self,
        stream: Stream,
        is_static: bool,
        message: TFMessage,
        observation: TransformBatchObservation,
    ) -> None:
        del stream, is_static, message
        if observation.input_error or not observation.required_chain_available:
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
            return
        self._try_process_pending_depth()

    def _after_image(
        self,
        stream: Stream,
        message: Image,
        observation: ImageObservation,
    ) -> None:
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
            return
        try:
            values = _depth_values(message)
        except ValueError as error:
            self.get_logger().warning(
                f"depth geometry rejected frame: {error}"
            )
            return
        if self._pending_depth is not None:
            self._pending_depth_drop_count += 1
        self._pending_depth = (observation.sensor_stamp_ns, values)
        self._try_process_pending_depth()

    def _try_process_pending_depth(self) -> None:
        if self._pending_depth is None:
            return
        stamp_ns, values = self._pending_depth
        if self._pipeline.alignment_at(stamp_ns) is None:
            return
        self._pending_depth = None
        result = self._pipeline.process_depth(stamp_ns, values)
        if result is None:
            return
        self._last_ground = result.ground
        self._last_candidate_count = len(result.candidates)
        if not result.confirmed:
            self._publish_health()
            return
        self._confirmed_observation_count += len(result.confirmed)
        self._latest_confirmation_spread_m = max(
            confirmed.spatial_spread_m for confirmed in result.confirmed
        )
        for confirmed in result.confirmed:
            self._cloud_publisher.publish(
                _point_cloud(confirmed.points_odom, stamp_ns)
            )
            self._cloud_publish_count += 1
        self._publish_health()

    def _operational_hazard_output_enabled(self) -> bool:
        return True

    def _effective_health_state(self, snapshot: HealthSnapshot) -> HealthState:
        if snapshot.state is HealthState.INVALID:
            return HealthState.INVALID
        if self._last_ground is not None and not self._last_ground.accepted:
            return HealthState.DEGRADED
        return snapshot.state

    def _additional_health_reasons(self) -> tuple[str, ...]:
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
                "ground.nominal_height_error_m": (
                    metrics.nominal_height_error_m
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
                "geometry.pending_depth_drops": (
                    self._pending_depth_drop_count
                ),
                "geometry.alignment_frame": "odom",
                "geometry.confirmation_observations": 2,
                "odom.maximum_interpolation_gap_ms": (
                    self._pipeline.odom.maximum_interpolation_gap_ns
                    / 1_000_000
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


def _point_cloud(
    points: tuple[tuple[float, float, float], ...], sensor_stamp_ns: int
) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.stamp = Time(nanoseconds=sensor_stamp_ns).to_msg()
    cloud.header.frame_id = "odom"
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = b"".join(struct.pack("<fff", *point) for point in points)
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
