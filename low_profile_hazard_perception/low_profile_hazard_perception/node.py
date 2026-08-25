"""ROS 2 adapter for independent RGB-D input health."""

from __future__ import annotations

import time
from functools import partial

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

try:
    from rclpy.qos_event import SubscriptionEventCallbacks
except ImportError:  # Jazzy and newer renamed the module.
    from rclpy.event_handler import SubscriptionEventCallbacks

from .health import (
    CameraInfoObservation,
    HealthMonitor,
    HealthSnapshot,
    HealthState,
    ImageObservation,
    OdomObservation,
    Stream,
    Transform,
    TransformBatchObservation,
)


def _stamp_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _text(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


class InputHealthNode(Node):
    """Validate independent input streams and publish observable health."""

    def __init__(self) -> None:
        super().__init__("input_health")
        expected_width = self.declare_parameter("expected_width", 640).value
        expected_height = self.declare_parameter("expected_height", 360).value
        sensor_stale_ms = self.declare_parameter(
            "sensor_stale_after_ms", 500.0
        ).value
        receive_stale_ms = self.declare_parameter(
            "receive_stale_after_ms", 500.0
        ).value
        publish_period_ms = self.declare_parameter(
            "health_publish_period_ms", 200.0
        ).value
        self._camera_frame = self.declare_parameter(
            "camera_frame", "camera_1_color_optical_frame"
        ).value
        self._base_frame = self.declare_parameter(
            "base_frame", "base_footprint"
        ).value
        self._monitor = HealthMonitor(
            expected_width=int(expected_width),
            expected_height=int(expected_height),
            expected_camera_frame=str(self._camera_frame),
            sensor_stale_after_ns=int(float(sensor_stale_ms) * 1_000_000),
            receive_stale_after_ns=int(float(receive_stale_ms) * 1_000_000),
        )
        self._tf_buffer = Buffer(node=self)
        self._profile_logged = False

        health_topic = self.declare_parameter(
            "health_topic", "/low_profile_hazard_perception/health"
        ).value
        health_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._health_publisher = self.create_publisher(
            DiagnosticArray, health_topic, health_qos
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        tf_static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._subscriptions = [
            self.create_subscription(
                Image,
                self._topic("color_image_topic", "/camera_1/color/image_raw"),
                partial(self._on_image, Stream.COLOR_IMAGE),
                image_qos,
                event_callbacks=self._events(Stream.COLOR_IMAGE),
            ),
            self.create_subscription(
                CameraInfo,
                self._topic(
                    "color_camera_info_topic", "/camera_1/color/camera_info"
                ),
                partial(self._on_camera_info, Stream.COLOR_CAMERA_INFO),
                image_qos,
                event_callbacks=self._events(Stream.COLOR_CAMERA_INFO),
            ),
            self.create_subscription(
                Image,
                self._topic("depth_image_topic", "/camera_1/depth/image_raw"),
                partial(self._on_image, Stream.DEPTH_IMAGE),
                image_qos,
                event_callbacks=self._events(Stream.DEPTH_IMAGE),
            ),
            self.create_subscription(
                CameraInfo,
                self._topic(
                    "depth_camera_info_topic", "/camera_1/depth/camera_info"
                ),
                partial(self._on_camera_info, Stream.DEPTH_CAMERA_INFO),
                image_qos,
                event_callbacks=self._events(Stream.DEPTH_CAMERA_INFO),
            ),
            self.create_subscription(
                Odometry,
                self._topic("odom_topic", "/odom"),
                self._on_odom,
                state_qos,
                event_callbacks=self._events(Stream.ODOM),
            ),
            self.create_subscription(
                TFMessage,
                self._topic("tf_topic", "/tf"),
                partial(self._on_tf, Stream.TF, False),
                state_qos,
                event_callbacks=self._events(Stream.TF),
            ),
            self.create_subscription(
                TFMessage,
                self._topic("tf_static_topic", "/tf_static"),
                partial(self._on_tf, Stream.TF_STATIC, True),
                tf_static_qos,
                event_callbacks=self._events(Stream.TF_STATIC),
            ),
        ]
        self._health_timer = self.create_timer(
            float(publish_period_ms) / 1000.0, self._publish_health
        )
        self._publish_health()

    def _topic(self, parameter: str, default: str) -> str:
        return str(self.declare_parameter(parameter, default).value)

    def _events(self, stream: Stream) -> SubscriptionEventCallbacks:
        return SubscriptionEventCallbacks(
            message_lost=partial(self._on_message_lost, stream),
            use_default_callbacks=False,
        )

    def _on_message_lost(self, stream: Stream, event: object) -> None:
        count = max(0, int(event.total_count_change))
        self._monitor.record_queue_drops(stream, count)
        self._publish_health()

    def _on_image(self, stream: Stream, message: Image) -> None:
        self._monitor.observe_image(
            stream,
            ImageObservation(
                sensor_stamp_ns=_stamp_ns(message.header.stamp),
                receive_time_ns=time.monotonic_ns(),
                frame_id=message.header.frame_id,
                width=message.width,
                height=message.height,
                step=message.step,
                encoding=message.encoding,
                data_size=len(message.data),
            ),
        )
        self._publish_health()

    def _on_camera_info(self, stream: Stream, message: CameraInfo) -> None:
        self._monitor.observe_camera_info(
            stream,
            CameraInfoObservation(
                sensor_stamp_ns=_stamp_ns(message.header.stamp),
                receive_time_ns=time.monotonic_ns(),
                frame_id=message.header.frame_id,
                width=message.width,
                height=message.height,
                fx=message.k[0],
                fy=message.k[4],
                cx=message.k[2],
                cy=message.k[5],
            ),
        )
        self._publish_health()

    def _on_odom(self, message: Odometry) -> None:
        self._monitor.observe_odom(
            OdomObservation(
                sensor_stamp_ns=_stamp_ns(message.header.stamp),
                receive_time_ns=time.monotonic_ns(),
                frame_id=message.header.frame_id,
                child_frame_id=message.child_frame_id,
            )
        )
        self._publish_health()

    def _on_tf(
        self, stream: Stream, is_static: bool, message: TFMessage
    ) -> None:
        receive_time_ns = time.monotonic_ns()
        for transform in message.transforms:
            if is_static:
                self._tf_buffer.set_transform_static(transform, "input_health")
            else:
                self._tf_buffer.set_transform(transform, "input_health")
        stamps = [
            _stamp_ns(transform.header.stamp)
            for transform in message.transforms
        ]
        sensor_stamp_ns = max(
            stamps, default=self.get_clock().now().nanoseconds
        )
        chain_available = self._tf_buffer.can_transform(
            self._base_frame, self._camera_frame, Time()
        )
        self._monitor.observe_transforms(
            stream,
            TransformBatchObservation(
                sensor_stamp_ns=sensor_stamp_ns,
                receive_time_ns=receive_time_ns,
                transforms=tuple(
                    Transform(
                        parent_frame_id=transform.header.frame_id,
                        child_frame_id=transform.child_frame_id,
                        translation=(
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z,
                        ),
                        rotation=(
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w,
                        ),
                    )
                    for transform in message.transforms
                ),
                required_chain_available=chain_available,
            ),
        )
        self._publish_health()

    def _snapshot(self) -> HealthSnapshot:
        return self._monitor.snapshot(
            sensor_now_ns=self.get_clock().now().nanoseconds,
            receive_now_ns=time.monotonic_ns(),
        )

    def _publish_health(self) -> None:
        snapshot = self._snapshot()
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "low_profile_hazard_perception/input_health"
        status.hardware_id = "dcw2"
        status.level = {
            HealthState.HEALTHY: DiagnosticStatus.OK,
            HealthState.DEGRADED: DiagnosticStatus.WARN,
            HealthState.INVALID: DiagnosticStatus.ERROR,
        }[snapshot.state]
        status.message = snapshot.state.value
        values: dict[str, object] = {
            "state": snapshot.state.value,
            "reasons": ",".join(snapshot.reasons),
            "missing_streams": ",".join(
                stream.value for stream in snapshot.missing_streams
            ),
            "stale_sensor_streams": ",".join(
                stream.value for stream in snapshot.stale_sensor_streams
            ),
            "stale_receive_streams": ",".join(
                stream.value for stream in snapshot.stale_receive_streams
            ),
            "invalid_streams": ",".join(
                stream.value for stream in snapshot.invalid_streams
            ),
            "tf_chain_available": snapshot.tf_chain_available,
            "camera_info.color.consistent": snapshot.camera_info_consistency[
                "color"
            ],
            "camera_info.depth.consistent": snapshot.camera_info_consistency[
                "depth"
            ],
            "sensor_age_clock": "ros_clock",
            "receive_age_clock": "steady_monotonic",
            "queue_drop_source": "rmw_message_lost",
            "operational_hazard_output_enabled": False,
        }
        for stream, health in snapshot.streams.items():
            prefix = stream.value
            values.update(
                {
                    f"{prefix}.delivered_count": health.delivered_count,
                    f"{prefix}.valid_count": health.valid_count,
                    f"{prefix}.invalid_count": health.invalid_count,
                    f"{prefix}.queue_drops": health.queue_drops,
                    f"{prefix}.sensor_stamp_age_ms": (
                        health.sensor_stamp_age_ms
                    ),
                    f"{prefix}.receive_age_ms": health.receive_age_ms,
                    f"{prefix}.approximate_rate_hz": (
                        health.approximate_rate_hz
                    ),
                    f"{prefix}.frame_id": health.frame_id,
                    f"{prefix}.profile": health.profile,
                    f"{prefix}.encoding": health.encoding,
                    f"{prefix}.valid": health.valid,
                    f"{prefix}.reason": health.reason,
                }
            )
        status.values = [
            KeyValue(key=key, value=_text(value))
            for key, value in values.items()
        ]
        message.status = [status]
        self._health_publisher.publish(message)
        self._log_profile_once(snapshot)

    def _log_profile_once(self, snapshot: HealthSnapshot) -> None:
        if self._profile_logged:
            return
        color = snapshot.streams[Stream.COLOR_IMAGE]
        depth = snapshot.streams[Stream.DEPTH_IMAGE]
        if not color.profile or not depth.profile:
            return
        self.get_logger().info(
            "Delivered RGB-D profile: "
            f"color={color.profile}/{color.encoding}, "
            f"depth={depth.profile}/{depth.encoding}"
        )
        self._profile_logged = True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InputHealthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
