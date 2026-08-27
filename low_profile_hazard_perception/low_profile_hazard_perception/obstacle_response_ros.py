"""ROS 2 subscriptions for the provider-independent obstacle-response seam."""

from __future__ import annotations

import time

from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField

from .obstacle_response import (
    ConfirmedCloudView,
    ObstacleResponseConfig,
    UnifiedObstacleResponseBridge,
    UnifiedObstacleResponsePort,
    decode_confirmed_cloud,
    perception_health_from_values,
)


_DATATYPE_NAMES = {
    PointField.INT8: "int8",
    PointField.UINT8: "uint8",
    PointField.INT16: "int16",
    PointField.UINT16: "uint16",
    PointField.INT32: "int32",
    PointField.UINT32: "uint32",
    PointField.FLOAT32: "float32",
    PointField.FLOAT64: "float64",
}


def _stamp_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ObstacleResponseAdapterNode(Node):
    """Consume two operational topics and invoke an injected robot port."""

    def __init__(
        self,
        *,
        port: UnifiedObstacleResponsePort,
        config: ObstacleResponseConfig,
    ) -> None:
        super().__init__("low_profile_hazard_response_adapter")
        self._bridge = UnifiedObstacleResponseBridge(config, port)
        operational_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._subscriptions = (
            self.create_subscription(
                PointCloud2,
                "/low_profile_hazard_perception/confirmed_hazards",
                self._on_cloud,
                operational_qos,
            ),
            self.create_subscription(
                DiagnosticArray,
                "/low_profile_hazard_perception/health",
                self._on_health,
                operational_qos,
            ),
        )
        check_period = min(config.health_timeout_ns / 2_000_000_000, 0.1)
        self._health_timer = self.create_timer(check_period, self._check_health)

    def _on_health(self, message: DiagnosticArray) -> None:
        statuses = [
            status
            for status in message.status
            if status.name == "low_profile_hazard_perception/input_health"
        ]
        if len(statuses) != 1:
            return
        status = statuses[0]
        values = {item.key: item.value for item in status.values}
        try:
            health = perception_health_from_values(
                state=status.message,
                values=values,
                received_monotonic_ns=time.monotonic_ns(),
                heartbeat_stamp_ns=_stamp_ns(message.header.stamp),
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            self._bridge.reject_operational_input("health_invalid")
            return
        self._bridge.consume_health(health)

    def _on_cloud(self, message: PointCloud2) -> None:
        fields = {
            field.name: (
                int(field.offset),
                _DATATYPE_NAMES.get(int(field.datatype), "unknown"),
            )
            for field in message.fields
        }
        try:
            snapshot = decode_confirmed_cloud(
                ConfirmedCloudView(
                    frame_id=message.header.frame_id,
                    header_stamp_ns=_stamp_ns(message.header.stamp),
                    width=int(message.width),
                    height=int(message.height),
                    point_step=int(message.point_step),
                    row_step=int(message.row_step),
                    data=bytes(message.data),
                    fields=fields,
                    is_bigendian=bool(message.is_bigendian),
                )
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            self._bridge.reject_operational_input("confirmed_cloud_invalid")
            return
        self._bridge.consume_cloud(
            snapshot,
            sensor_now_ns=self.get_clock().now().nanoseconds,
            received_monotonic_ns=time.monotonic_ns(),
        )

    def _check_health(self) -> None:
        self._bridge.tick(received_monotonic_ns=time.monotonic_ns())
