"""Black-box ROS graph checks for the ticket-1 public seam."""

import time
from collections.abc import Callable

import pytest


rclpy = pytest.importorskip("rclpy")

from diagnostic_msgs.msg import DiagnosticArray  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from low_profile_hazard_perception.node import InputHealthNode  # noqa: E402


def _qos(*, transient: bool = False, depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=(
            ReliabilityPolicy.RELIABLE
            if transient
            else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if transient
            else DurabilityPolicy.VOLATILE
        ),
    )


def _values(message: DiagnosticArray) -> dict[str, str]:
    return {item.key: item.value for item in message.status[0].values}


def _spin_until(
    executor: SingleThreadedExecutor,
    predicate: Callable[[], bool],
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS graph observation")


def _image(stamp: object, *, depth: bool, height: int = 360) -> Image:
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "camera_1_color_optical_frame"
    message.width = 640
    message.height = height
    bytes_per_pixel = 2 if depth else 3
    message.encoding = "16UC1" if depth else "rgb8"
    message.step = 640 * bytes_per_pixel
    message.data = bytes(message.step * height)
    return message


def _camera_info(stamp: object) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = "camera_1_color_optical_frame"
    message.width = 640
    message.height = 360
    message.k = [455.0, 0.0, 320.0, 0.0, 455.0, 180.0, 0.0, 0.0, 1.0]
    return message


def _transform(stamp: object, parent: str, child: str) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.rotation.w = 1.0
    return message


def test_independent_inputs_drive_health_without_hazard_output() -> None:
    rclpy.init()
    health_node = InputHealthNode()
    driver = Node("health_test_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(health_node)
    executor.add_node(driver)
    latest: list[DiagnosticArray] = []
    health_qos = _qos(transient=True, depth=1)
    driver.create_subscription(
        DiagnosticArray,
        "/low_profile_hazard_perception/health",
        latest.append,
        health_qos,
    )
    publishers = {
        "color": driver.create_publisher(
            Image, "/camera_1/color/image_raw", _qos(depth=1)
        ),
        "color_info": driver.create_publisher(
            CameraInfo, "/camera_1/color/camera_info", _qos(depth=1)
        ),
        "depth": driver.create_publisher(
            Image, "/camera_1/depth/image_raw", _qos(depth=1)
        ),
        "depth_info": driver.create_publisher(
            CameraInfo, "/camera_1/depth/camera_info", _qos(depth=1)
        ),
        "odom": driver.create_publisher(Odometry, "/odom", _qos()),
        "tf": driver.create_publisher(TFMessage, "/tf", _qos()),
        "tf_static": driver.create_publisher(
            TFMessage, "/tf_static", _qos(transient=True, depth=100)
        ),
    }
    try:
        _spin_until(executor, lambda: bool(latest))
        stamp = driver.get_clock().now().to_msg()
        publishers["color"].publish(_image(stamp, depth=False))
        _spin_until(
            executor,
            lambda: _values(latest[-1]).get("color_image.delivered_count")
            == "1",
        )
        values = _values(latest[-1])
        assert values["depth_image.delivered_count"] == "0"

        publishers["color_info"].publish(_camera_info(stamp))
        publishers["depth"].publish(_image(stamp, depth=True))
        publishers["depth_info"].publish(_camera_info(stamp))
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        publishers["odom"].publish(odom)
        publishers["tf"].publish(
            TFMessage(transforms=[_transform(stamp, "odom", "base_footprint")])
        )
        publishers["tf_static"].publish(
            TFMessage(
                transforms=[
                    _transform(
                        stamp,
                        "base_footprint",
                        "camera_1_color_optical_frame",
                    )
                ]
            )
        )
        _spin_until(
            executor,
            lambda: _values(latest[-1]).get("state") == "HEALTHY",
        )
        values = _values(latest[-1])
        assert values["camera_info.color.consistent"] == "true"
        assert values["camera_info.depth.consistent"] == "true"
        assert values["operational_hazard_output_enabled"] == "false"
        assert values["depth_image.processing_latency_ms"] != "unknown"
        topic_types = dict(health_node.get_topic_names_and_types())
        assert all(
            "sensor_msgs/msg/PointCloud2" not in types
            for types in topic_types.values()
        )

        invalid_tf = _transform(stamp, "odom", "base_footprint")
        invalid_tf.transform.rotation.w = 0.0
        publishers["tf"].publish(TFMessage(transforms=[invalid_tf]))
        _spin_until(
            executor,
            lambda: "zero norm" in _values(latest[-1]).get("tf.reason", ""),
        )
        assert _values(latest[-1])["state"] == "INVALID"

        publishers["tf"].publish(
            TFMessage(transforms=[_transform(stamp, "odom", "base_footprint")])
        )
        _spin_until(
            executor,
            lambda: _values(latest[-1]).get("state") == "HEALTHY",
        )

        publishers["depth"].publish(_image(stamp, depth=True, height=400))
        _spin_until(
            executor,
            lambda: _values(latest[-1]).get("state") == "INVALID",
        )
        assert "invalid:depth_image" in _values(latest[-1])["reasons"]
    finally:
        executor.remove_node(driver)
        executor.remove_node(health_node)
        driver.destroy_node()
        health_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
