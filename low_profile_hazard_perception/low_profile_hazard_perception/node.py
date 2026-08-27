"""ROS 2 adapter for independent RGB-D input health."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
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
    validate_transform_batch,
)
from .detection_profile import (
    DetectionProfile,
    default_detection_profile_path,
)
from .latest_input_queue import LatestInputQueue
from .resource_budget import ProcessResourceMonitor


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

    def __init__(self, *, bind_detection_profile: bool = False) -> None:
        super().__init__("input_health")
        self._detection_profile: DetectionProfile | None = None
        self._operating_speed_mps: float | None = None
        if bind_detection_profile:
            profile_path = str(
                self.declare_parameter(
                    "detection_profile_path",
                    str(default_detection_profile_path()),
                ).value
            )
            self._detection_profile = DetectionProfile.load(profile_path)
            self._operating_speed_mps = float(
                self.declare_parameter("operating_speed_mps", 0.3).value
            )
            self._detection_profile.validate_speed(self._operating_speed_mps)
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
        processing_period_ms = self.declare_parameter(
            "input_processing_period_ms", 5.0
        ).value
        self._camera_frame = self.declare_parameter(
            "camera_frame", "camera_1_color_optical_frame"
        ).value
        self._base_frame = self.declare_parameter(
            "base_frame", "base_footprint"
        ).value
        if self._detection_profile is not None:
            profile_runtime = {
                    "expected_width": expected_width,
                    "expected_height": expected_height,
                    "camera_frame": self._camera_frame,
                    "operating_speed_mps": self._operating_speed_mps,
            }
            self._detection_profile.validate_parameters(
                profile_runtime,
                names=tuple(profile_runtime),
            )
        self._monitor = HealthMonitor(
            expected_width=int(expected_width),
            expected_height=int(expected_height),
            expected_camera_frame=str(self._camera_frame),
            sensor_stale_after_ns=int(float(sensor_stale_ms) * 1_000_000),
            receive_stale_after_ns=int(float(receive_stale_ms) * 1_000_000),
        )
        self._tf_buffer = Buffer(node=self)
        self._profile_logged = False
        self._input_queues: dict[
            Stream, LatestInputQueue[Callable[[], int]]
        ] = {stream: LatestInputQueue() for stream in Stream}
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._resource_monitor = ProcessResourceMonitor(
            capacity=720,
            npu_state=(
                "disabled_rule_profile"
                if self._detection_profile is not None
                and self._detection_profile.model_version == "none"
                else "not_configured"
            ),
        )

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
        self._processing_timer = self.create_timer(
            float(processing_period_ms) / 1000.0,
            self._drain_input_queues,
            clock=self._steady_clock,
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
        self._monitor.record_transport_drops(stream, count)
        self._publish_health()

    def _enqueue(
        self,
        stream: Stream,
        work: Callable[[], int],
        *,
        sensor_stamp_ns: int,
    ) -> None:
        if self._input_queues[stream].offer(
            work, sensor_stamp_ns=sensor_stamp_ns
        ):
            self._monitor.record_queue_drops(stream, 1)
            self._publish_health()

    def _drain_input_queues(self) -> None:
        processed = False
        for stream, queue in self._input_queues.items():
            work = queue.take()
            if work is None:
                continue
            receive_time_ns = work()
            self._monitor.record_processing_complete(
                stream,
                receive_time_ns=receive_time_ns,
                processing_complete_time_ns=time.monotonic_ns(),
            )
            processed = True
        if processed:
            self._publish_health()

    def _process_immediately(
        self,
        stream: Stream,
        work: Callable[[], int],
        *,
        sensor_stamp_ns: int,
    ) -> None:
        queue = self._input_queues[stream]
        if queue.offer(work, sensor_stamp_ns=sensor_stamp_ns):
            self._monitor.record_queue_drops(stream, 1)
        pending = queue.take()
        if pending is None:
            return
        receive_time_ns = pending()
        self._monitor.record_processing_complete(
            stream,
            receive_time_ns=receive_time_ns,
            processing_complete_time_ns=time.monotonic_ns(),
        )
        self._publish_health()

    def _on_image(self, stream: Stream, message: Image) -> None:
        observation = ImageObservation(
            sensor_stamp_ns=_stamp_ns(message.header.stamp),
            receive_time_ns=time.monotonic_ns(),
            frame_id=message.header.frame_id,
            width=message.width,
            height=message.height,
            step=message.step,
            encoding=message.encoding,
            data_size=len(message.data),
        )
        self._enqueue(
            stream,
            partial(self._process_image, stream, observation, message),
            sensor_stamp_ns=observation.sensor_stamp_ns,
        )

    def _process_image(
        self,
        stream: Stream,
        observation: ImageObservation,
        message: Image,
    ) -> int:
        self._monitor.observe_image(stream, observation)
        self._after_image(stream, message, observation)
        return observation.receive_time_ns

    def _after_image(
        self,
        stream: Stream,
        message: Image,
        observation: ImageObservation,
    ) -> None:
        """Extension seam for independently processed image evidence."""

    def _on_camera_info(self, stream: Stream, message: CameraInfo) -> None:
        observation = CameraInfoObservation(
            sensor_stamp_ns=_stamp_ns(message.header.stamp),
            receive_time_ns=time.monotonic_ns(),
            frame_id=message.header.frame_id,
            width=message.width,
            height=message.height,
            fx=message.k[0],
            fy=message.k[4],
            cx=message.k[2],
            cy=message.k[5],
        )
        self._enqueue(
            stream,
            partial(self._process_camera_info, stream, observation),
            sensor_stamp_ns=observation.sensor_stamp_ns,
        )

    def _process_camera_info(
        self, stream: Stream, observation: CameraInfoObservation
    ) -> int:
        self._monitor.observe_camera_info(stream, observation)
        self._after_camera_info(stream, observation)
        return observation.receive_time_ns

    def _after_camera_info(
        self, stream: Stream, observation: CameraInfoObservation
    ) -> None:
        """Extension seam for a validated per-stream camera model."""

    def _on_odom(self, message: Odometry) -> None:
        self._before_odom(message)
        observation = OdomObservation(
            sensor_stamp_ns=_stamp_ns(message.header.stamp),
            receive_time_ns=time.monotonic_ns(),
            frame_id=message.header.frame_id,
            child_frame_id=message.child_frame_id,
        )
        self._enqueue(
            Stream.ODOM,
            partial(self._process_odom, observation),
            sensor_stamp_ns=observation.sensor_stamp_ns,
        )

    def _before_odom(self, message: Odometry) -> None:
        """Keep every odom pose available to extension interpolation."""

    def _process_odom(self, observation: OdomObservation) -> int:
        self._monitor.observe_odom(observation)
        self._after_odom(observation)
        return observation.receive_time_ns

    def _after_odom(self, observation: OdomObservation) -> None:
        """Extension seam after odom health has observed the sample."""

    def _on_tf(
        self, stream: Stream, is_static: bool, message: TFMessage
    ) -> None:
        stamps = [
            _stamp_ns(transform.header.stamp)
            for transform in message.transforms
        ]
        sensor_stamp_ns = max(
            stamps, default=self.get_clock().now().nanoseconds
        )
        work = partial(
            self._process_tf,
            stream,
            is_static,
            message,
            time.monotonic_ns(),
            sensor_stamp_ns,
        )
        if is_static:
            self._process_immediately(
                stream, work, sensor_stamp_ns=sensor_stamp_ns
            )
        else:
            self._enqueue(stream, work, sensor_stamp_ns=sensor_stamp_ns)

    def _process_tf(
        self,
        stream: Stream,
        is_static: bool,
        message: TFMessage,
        receive_time_ns: int,
        sensor_stamp_ns: int,
    ) -> int:
        observation = TransformBatchObservation(
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
            required_chain_available=bool(
                self._tf_buffer.can_transform(
                    self._base_frame, self._camera_frame, Time()
                )
            ),
        )
        if not validate_transform_batch(stream, observation):
            try:
                for transform in message.transforms:
                    if is_static:
                        self._tf_buffer.set_transform_static(
                            transform, "input_health"
                        )
                    else:
                        self._tf_buffer.set_transform(
                            transform, "input_health"
                        )
            except Exception as error:  # tf2 supplies the diagnostic text.
                observation = replace(
                    observation,
                    input_error=f"TF buffer rejected transform: {error}",
                )
        observation = replace(
            observation,
            required_chain_available=bool(
                self._tf_buffer.can_transform(
                    self._base_frame, self._camera_frame, Time()
                )
            ),
        )
        self._monitor.observe_transforms(stream, observation)
        self._after_tf(stream, is_static, message, observation)
        return receive_time_ns

    def _after_tf(
        self,
        stream: Stream,
        is_static: bool,
        message: TFMessage,
        observation: TransformBatchObservation,
    ) -> None:
        """Extension seam called after accepted transforms enter tf2."""

    def _snapshot(self) -> HealthSnapshot:
        return self._monitor.snapshot(
            sensor_now_ns=self.get_clock().now().nanoseconds,
            receive_now_ns=time.monotonic_ns(),
        )

    def _publish_health(self) -> None:
        snapshot = self._snapshot()
        effective_state = self._effective_health_state(snapshot)
        reasons = snapshot.reasons + self._additional_health_reasons()
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "low_profile_hazard_perception/input_health"
        status.hardware_id = "dcw2"
        status.level = {
            HealthState.HEALTHY: DiagnosticStatus.OK,
            HealthState.DEGRADED: DiagnosticStatus.WARN,
            HealthState.INVALID: DiagnosticStatus.ERROR,
        }[effective_state]
        status.message = effective_state.value
        values: dict[str, object] = {
            "state": effective_state.value,
            "reasons": ",".join(reasons),
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
            "queue_drop_source": "latest_input_queue",
            "transport_drop_source": "rmw_message_lost",
            "operational_hazard_output_enabled": (
                self._operational_hazard_output_enabled()
            ),
        }
        values.update(self._resource_monitor.sample().diagnostic_values())
        if self._detection_profile is not None:
            profile = self._detection_profile
            values.update(
                {
                    "profile.id": profile.profile_id,
                    "profile.schema_version": profile.schema_version,
                    "profile.fingerprint": profile.fingerprint,
                    "profile.validation_phase": profile.validation_phase,
                    "profile.image": profile.camera.image_profile,
                    "profile.minimum_rate_hz": (
                        profile.camera.validated_rate_range_hz[0]
                    ),
                    "profile.maximum_rate_hz": (
                        profile.camera.validated_rate_range_hz[1]
                    ),
                    "profile.rgb_encoding": profile.camera.rgb_encoding,
                    "profile.depth_encoding": profile.camera.depth_encoding,
                    "profile.camera_frame": profile.camera.frame_id,
                    "profile.observed_camera_height_m": (
                        profile.observed_camera_height_m
                    ),
                    "profile.minimum_camera_height_m": (
                        profile.validated_height_range_m[0]
                    ),
                    "profile.maximum_camera_height_m": (
                        profile.validated_height_range_m[1]
                    ),
                    "profile.observed_downward_pitch_degrees": (
                        profile.observed_downward_pitch_degrees
                    ),
                    "profile.minimum_downward_pitch_degrees": (
                        profile.validated_downward_pitch_range_degrees[0]
                    ),
                    "profile.maximum_downward_pitch_degrees": (
                        profile.validated_downward_pitch_range_degrees[1]
                    ),
                    "profile.footprint_minimum_x_m": (
                        profile.footprint_m["minimum_x"]
                    ),
                    "profile.footprint_maximum_x_m": (
                        profile.footprint_m["maximum_x"]
                    ),
                    "profile.footprint_minimum_y_m": (
                        profile.footprint_m["minimum_y"]
                    ),
                    "profile.footprint_maximum_y_m": (
                        profile.footprint_m["maximum_y"]
                    ),
                    "profile.rule_version": profile.rule_version,
                    "profile.model_version": profile.model_version,
                    "profile.maximum_speed_mps": profile.maximum_speed_mps,
                    "profile.configured_operating_speed_mps": (
                        self._operating_speed_mps
                    ),
                    "budget.processing_p95_ms": (
                        profile.resource_budget.processing_p95_ms
                    ),
                    "budget.depth_geometry_average_cpu_cores": (
                        profile.resource_budget.depth_geometry_average_cpu_cores
                    ),
                    "budget.soak_duration_seconds": (
                        profile.resource_budget.soak_duration_seconds
                    ),
                    "budget.maximum_input_queue_depth": (
                        profile.resource_budget.maximum_input_queue_depth
                    ),
                    "budget.maximum_rgb_reorder_depth": (
                        profile.resource_budget.maximum_rgb_reorder_depth
                    ),
                    "budget.maximum_memory_growth_bytes": (
                        profile.resource_budget.maximum_memory_growth_bytes
                    ),
                }
            )
        values.update(self._additional_health_values())
        for stream, health in snapshot.streams.items():
            prefix = stream.value
            queue = self._input_queues[stream]
            values.update(
                {
                    f"{prefix}.delivered_count": queue.received_count,
                    f"{prefix}.processed_count": queue.processed_count,
                    f"{prefix}.pending_count": queue.pending_count,
                    f"{prefix}.valid_count": health.valid_count,
                    f"{prefix}.invalid_count": health.invalid_count,
                    f"{prefix}.queue_drops": health.queue_drops,
                    f"{prefix}.transport_drops": health.transport_drops,
                    f"{prefix}.sensor_stamp_age_ms": (
                        health.sensor_stamp_age_ms
                    ),
                    f"{prefix}.receive_age_ms": health.receive_age_ms,
                    f"{prefix}.processing_latency_ms": (
                        health.processing_latency_ms
                    ),
                    f"{prefix}.approximate_rate_hz": (
                        queue.approximate_received_rate_hz
                    ),
                    f"{prefix}.processed_rate_hz": (
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

    def _operational_hazard_output_enabled(self) -> bool:
        return False

    def _additional_health_values(self) -> dict[str, object]:
        return {}

    def _effective_health_state(self, snapshot: HealthSnapshot) -> HealthState:
        return snapshot.state

    def _additional_health_reasons(self) -> tuple[str, ...]:
        return ()


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
