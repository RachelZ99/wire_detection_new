"""Black-box ROS graph check for the ticket-2 operational seam."""

from array import array
import math
import struct
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
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from low_profile_hazard_perception.geometric_node import (  # noqa: E402
    GeometricHazardNode,
)


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


def _spin_until(
    executor: SingleThreadedExecutor,
    predicate: Callable[[], bool],
    timeout: float = 4.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS graph observation")


def _values(messages: list[DiagnosticArray]) -> dict[str, str]:
    return {item.key: item.value for item in messages[-1].status[0].values}


def _camera_info(stamp: object) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = "camera_1_color_optical_frame"
    message.width = 640
    message.height = 360
    message.k = [455.0, 0.0, 320.0, 0.0, 455.0, 180.0, 0.0, 0.0, 1.0]
    return message


def _color(stamp: object) -> Image:
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "camera_1_color_optical_frame"
    message.width = 640
    message.height = 360
    message.encoding = "rgb8"
    message.step = 640 * 3
    message.data = bytes(message.step * message.height)
    return message


def _depth(stamp: object, *, object_center_column: int) -> Image:
    width = 640
    height = 360
    fx = fy = 455.0
    cx = 320.0
    cy = 180.0
    pitch = math.radians(3.0)
    normal = (0.0, -math.cos(pitch), -math.sin(pitch))
    values = array("H")
    for row in range(height):
        for column in range(width):
            ray = ((column - cx) / fx, (row - cy) / fy, 1.0)
            denominator = sum(
                left * right for left, right in zip(normal, ray, strict=True)
            )
            if denominator >= -1e-6:
                values.append(0)
                continue
            raised = (
                255 <= row < 282
                and object_center_column - 30
                <= column
                < object_center_column + 30
            )
            height_above_ground = 0.030 if raised else 0.0
            depth_m = (height_above_ground - 0.225) / denominator
            values.append(max(0, min(65535, int(round(depth_m * 1000.0)))))
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "camera_1_color_optical_frame"
    message.width = width
    message.height = height
    message.encoding = "16UC1"
    message.is_bigendian = False
    message.step = width * 2
    message.data = values.tobytes()
    return message


def _invalid_depth(stamp: object) -> Image:
    message = _depth(stamp, object_center_column=360)
    message.data = bytes(message.step * message.height)
    return message


def _odom(stamp: object, x: float) -> Odometry:
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = "odom"
    message.child_frame_id = "base_footprint"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    return message


def _transform(
    stamp: object, parent: str, child: str, *, x: float = 0.0, z: float = 0.0
) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.translation.x = x
    message.transform.translation.z = z
    message.transform.rotation.w = 1.0
    return message


def test_two_motion_aligned_depth_observations_publish_one_odom_cloud() -> (
    None
):
    rclpy.init()
    perception = GeometricHazardNode()
    driver = Node("geometric_hazard_test_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(perception)
    executor.add_node(driver)
    health: list[DiagnosticArray] = []
    clouds: list[PointCloud2] = []
    driver.create_subscription(
        DiagnosticArray,
        "/low_profile_hazard_perception/health",
        health.append,
        _qos(transient=True, depth=1),
    )
    driver.create_subscription(
        PointCloud2,
        "/low_profile_hazard_perception/confirmed_hazards",
        clouds.append,
        QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        ),
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
        _spin_until(executor, lambda: bool(health))
        base_ns = driver.get_clock().now().nanoseconds + 100_000_000
        first_stamp = Time(nanoseconds=base_ns + 100_000_000).to_msg()
        second_stamp = Time(nanoseconds=base_ns + 300_000_000).to_msg()
        publishers["color_info"].publish(_camera_info(first_stamp))
        publishers["depth_info"].publish(_camera_info(first_stamp))
        publishers["color"].publish(_color(first_stamp))
        publishers["tf_static"].publish(
            TFMessage(
                transforms=[
                    _transform(
                        first_stamp,
                        "base_footprint",
                        "camera_1_color_optical_frame",
                        x=0.33,
                        z=0.15,
                    )
                ]
            )
        )
        publishers["tf"].publish(
            TFMessage(
                transforms=[_transform(first_stamp, "odom", "base_footprint")]
            )
        )
        odom_samples = (
            (base_ns + 50_000_000, 0.0),
            (base_ns + 150_000_000, 0.0),
            (base_ns + 250_000_000, 0.1),
            (base_ns + 350_000_000, 0.1),
        )
        for stamp_ns, x in odom_samples:
            publishers["odom"].publish(
                _odom(Time(nanoseconds=stamp_ns).to_msg(), x)
            )
            executor.spin_once(timeout_sec=0.02)

        publishers["depth"].publish(
            _depth(first_stamp, object_center_column=360)
        )
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.latest_candidate_count")
            == "1",
        )
        assert clouds == []

        publishers["depth"].publish(
            _depth(second_stamp, object_center_column=302)
        )
        _spin_until(executor, lambda: len(clouds) == 1)
        cloud = clouds[0]
        assert cloud.header.frame_id == "odom"
        assert cloud.header.stamp == second_stamp
        assert cloud.width > 100
        spread_field = next(
            field
            for field in cloud.fields
            if field.name == "confirmation_spread"
        )
        confirmation_spread = struct.unpack_from(
            "<f", bytes(cloud.data), spread_field.offset
        )[0]
        assert confirmation_spread < 0.025
        points = [
            struct.unpack_from("<fff", bytes(cloud.data), offset)
            for offset in range(0, len(cloud.data), cloud.point_step)
        ]
        centroid_x = sum(point[0] for point in points) / len(points)
        assert 0.35 < centroid_x < 0.55
        values = _values(health)
        assert values["ground.state"] == "VALID"
        assert 0.20 < float(values["ground.camera_height_m"]) < 0.25
        assert values["geometry.alignment_frame"] == "odom"
        assert values["geometry.confirmation_observations"] == "2"
        assert values["geometry.active_retained_hazard_count"] == "1"
        assert values["geometry.candidate_retention_ms"] == "500.000"
        assert values["geometry.confirmed_retention_ms"] == "2000.000"
        assert values["geometry.output_durability"] == "transient_local"

        failure_stamp = Time(nanoseconds=base_ns + 400_000_000).to_msg()
        publishers["odom"].publish(
            _odom(Time(nanoseconds=base_ns + 450_000_000).to_msg(), 0.1)
        )
        executor.spin_once(timeout_sec=0.02)
        publishers["depth"].publish(_invalid_depth(failure_stamp))
        _spin_until(
            executor,
            lambda: _values(health).get("ground.state") == "REJECTED",
        )
        values = _values(health)
        assert values["state"] == "DEGRADED"
        assert values["geometry.active_retained_hazard_count"] == "1"
        assert values["geometry.degradation_reason"].startswith("ground:")
        assert len(clouds) == 1

        recovery_stamp = Time(nanoseconds=base_ns + 500_000_000).to_msg()
        publishers["odom"].publish(
            _odom(Time(nanoseconds=base_ns + 550_000_000).to_msg(), 0.1)
        )
        executor.spin_once(timeout_sec=0.02)
        publishers["depth"].publish(
            _depth(recovery_stamp, object_center_column=302)
        )
        _spin_until(
            executor,
            lambda: len(clouds) == 2
            and _values(health).get("geometry.degradation_reason") == "",
        )
        recovered_points = [
            struct.unpack_from("<fff", bytes(clouds[-1].data), offset)
            for offset in range(
                0, len(clouds[-1].data), clouds[-1].point_step
            )
        ]
        recovered_centroid_x = sum(
            point[0] for point in recovered_points
        ) / len(recovered_points)
        assert abs(recovered_centroid_x - centroid_x) < 0.025
        values = _values(health)
        assert values["geometry.active_retained_hazard_count"] == "1"
        assert values["geometry.degradation_reason"] == ""
    finally:
        executor.remove_node(driver)
        executor.remove_node(perception)
        driver.destroy_node()
        perception.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
