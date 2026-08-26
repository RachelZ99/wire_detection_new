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
            ReliabilityPolicy.RELIABLE if transient else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL if transient else DurabilityPolicy.VOLATILE
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


def _cable_color(stamp: object, *, center_column: int) -> Image:
    message = _color(stamp)
    data = bytearray((82, 86, 88) * (message.width * message.height))
    for row in range(240, 321):
        for column in range(center_column - 1, center_column + 2):
            offset = (row * message.width + column) * 3
            data[offset : offset + 3] = bytes((235, 235, 235))
    message.data = bytes(data)
    return message


def _depth(stamp: object, *, object_center_column: int | None) -> Image:
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
                object_center_column is not None
                and 255 <= row < 282
                and object_center_column - 30 <= column < object_center_column + 30
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


def test_two_rgb_cable_observations_publish_the_same_odom_cloud() -> None:
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "sensor_stale_after_ms:=5000.0",
            "-p",
            "receive_stale_after_ms:=5000.0",
        ]
    )
    perception = GeometricHazardNode()
    driver = Node("rgb_cable_hazard_test_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(perception)
    executor.add_node(driver)
    health: list[DiagnosticArray] = []
    candidate_diagnostics: list[DiagnosticArray] = []
    clouds: list[PointCloud2] = []
    driver.create_subscription(
        DiagnosticArray,
        "/low_profile_hazard_perception/health",
        health.append,
        _qos(transient=True, depth=1),
    )
    driver.create_subscription(
        DiagnosticArray,
        "/low_profile_hazard_perception/candidate_diagnostics",
        candidate_diagnostics.append,
        _qos(depth=1),
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
        depth_stamp = Time(nanoseconds=base_ns + 200_000_000).to_msg()
        first_rgb_stamp = Time(nanoseconds=base_ns + 300_000_000).to_msg()
        second_rgb_stamp = Time(nanoseconds=base_ns + 500_000_000).to_msg()
        publishers["color_info"].publish(_camera_info(first_rgb_stamp))
        publishers["depth_info"].publish(_camera_info(depth_stamp))
        publishers["tf_static"].publish(
            TFMessage(
                transforms=[
                    _transform(
                        depth_stamp,
                        "base_footprint",
                        "camera_1_color_optical_frame",
                        x=0.33,
                        z=0.15,
                    )
                ]
            )
        )
        publishers["tf"].publish(
            TFMessage(transforms=[_transform(depth_stamp, "odom", "base_footprint")])
        )
        for stamp_ns in range(
            base_ns + 150_000_000,
            base_ns + 651_000_000,
            50_000_000,
        ):
            publishers["odom"].publish(_odom(Time(nanoseconds=stamp_ns).to_msg(), 0.0))
            executor.spin_once(timeout_sec=0.01)
        publishers["depth"].publish(_depth(depth_stamp, object_center_column=None))
        publishers["color"].publish(_cable_color(first_rgb_stamp, center_column=320))
        _spin_until(
            executor,
            lambda: _values(health).get("cable.processed_rgb_count") == "1"
            and _values(health).get("cable.latest_candidate_count") == "1",
            timeout=15.0,
        )
        assert clouds == []

        publishers["color"].publish(_cable_color(second_rgb_stamp, center_column=320))
        _spin_until(executor, lambda: bool(clouds), timeout=15.0)
        _spin_until(
            executor,
            lambda: bool(candidate_diagnostics)
            and _values(candidate_diagnostics).get("decision_reason")
            == "CONFIRMED_RGB_CABLE",
            timeout=4.0,
        )

        cloud = clouds[-1]
        assert cloud.header.frame_id == "odom"
        assert cloud.header.stamp == second_rgb_stamp
        evidence_field = next(
            field for field in cloud.fields if field.name == "evidence_mask"
        )
        evidence_masks = [
            struct.unpack_from("<B", bytes(cloud.data), offset + evidence_field.offset)[
                0
            ]
            for offset in range(0, len(cloud.data), cloud.point_step)
        ]
        assert evidence_masks
        assert all(mask & 2 for mask in evidence_masks)
        candidate_values = _values(candidate_diagnostics)
        assert candidate_values["evidence_sources"] == "RGB_CABLE"
        assert candidate_values["ground.accepted"] == "true"
        assert float(candidate_values["ground.inlier_ratio"]) > 0.65
        assert float(candidate_values["confidence"]) >= 0.75
        assert candidate_values["operational_confirmed"] == "true"
        values = _values(health)
        assert values["cable.provider"] == "training_free_thin_line"
        assert values["cable.confirmed_observation_count"] == "1"
        assert values["cable.confirmation_observations"] == "2"
        assert values["cable.rgb_depth_synchronizer"] == "disabled"
        assert values["cable.diagnostic_pink_operational"] == "false"
    finally:
        executor.remove_node(driver)
        executor.remove_node(perception)
        driver.destroy_node()
        perception.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


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


def test_two_motion_aligned_depth_observations_publish_one_odom_cloud() -> (None):
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "sensor_stale_after_ms:=5000.0",
            "-p",
            "receive_stale_after_ms:=5000.0",
        ]
    )
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
        third_stamp = Time(nanoseconds=base_ns + 400_000_000).to_msg()
        fourth_stamp = Time(nanoseconds=base_ns + 500_000_000).to_msg()
        publishers["depth"].publish(_depth(first_stamp, object_center_column=360))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason")
            == "camera_info:missing",
        )
        assert clouds == []

        publishers["color_info"].publish(_camera_info(first_stamp))
        publishers["depth_info"].publish(_camera_info(first_stamp))
        publishers["color"].publish(_color(first_stamp))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason") == "tf:missing",
        )
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
            TFMessage(transforms=[_transform(first_stamp, "odom", "base_footprint")])
        )
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason")
            == "odom:missing",
        )
        odom_samples = (
            (base_ns + 50_000_000, 0.0),
            (base_ns + 150_000_000, 0.0),
            (base_ns + 250_000_000, 0.1),
            (base_ns + 350_000_000, 0.1),
            (base_ns + 450_000_000, 0.1),
            (base_ns + 550_000_000, 0.1),
        )
        for stamp_ns, x in odom_samples:
            publishers["odom"].publish(_odom(Time(nanoseconds=stamp_ns).to_msg(), x))
            executor.spin_once(timeout_sec=0.02)

        _spin_until(
            executor,
            lambda: _values(health).get("geometry.latest_candidate_count") == "1",
        )
        assert clouds == []

        stale_stamp = Time(nanoseconds=base_ns - 6_000_000_000).to_msg()
        publishers["depth_info"].publish(_camera_info(stale_stamp))
        _spin_until(
            executor,
            lambda: float(
                _values(health).get("depth_camera_info.sensor_stamp_age_ms", "0")
            )
            > 5000.0,
        )
        publishers["depth"].publish(_depth(second_stamp, object_center_column=302))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason")
            == "camera_info:sensor_stale",
        )
        assert clouds == []
        publishers["depth_info"].publish(_camera_info(second_stamp))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.processed_depth_count") == "2",
        )

        publishers["tf"].publish(
            TFMessage(transforms=[_transform(stale_stamp, "odom", "base_footprint")])
        )
        _spin_until(
            executor,
            lambda: float(_values(health).get("tf.sensor_stamp_age_ms", "0")) > 5000.0,
        )
        publishers["depth"].publish(_depth(third_stamp, object_center_column=302))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason")
            == "tf:sensor_stale",
        )
        assert clouds == []
        publishers["tf"].publish(
            TFMessage(transforms=[_transform(third_stamp, "odom", "base_footprint")])
        )
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.processed_depth_count") == "3",
        )

        publishers["depth"].publish(_depth(fourth_stamp, object_center_column=302))
        _spin_until(executor, lambda: len(clouds) == 1)
        cloud = clouds[0]
        assert cloud.header.frame_id == "odom"
        assert cloud.header.stamp == fourth_stamp
        assert cloud.width > 100
        field_names = {field.name for field in cloud.fields}
        assert "observation_stamp_sec" in field_names
        assert "observation_stamp_nanosec" in field_names
        assert "evidence_mask" in field_names
        evidence_field = next(
            field for field in cloud.fields if field.name == "evidence_mask"
        )
        assert (
            struct.unpack_from("<B", bytes(cloud.data), evidence_field.offset)[0] == 1
        )
        stamp_sec_field = next(
            field for field in cloud.fields if field.name == "observation_stamp_sec"
        )
        stamp_nanosec_field = next(
            field for field in cloud.fields if field.name == "observation_stamp_nanosec"
        )
        point_source_stamp = (
            struct.unpack_from("<i", bytes(cloud.data), stamp_sec_field.offset)[0]
            * 1_000_000_000
            + struct.unpack_from("<I", bytes(cloud.data), stamp_nanosec_field.offset)[0]
        )
        assert point_source_stamp == (
            fourth_stamp.sec * 1_000_000_000 + fourth_stamp.nanosec
        )
        spread_field = next(
            field for field in cloud.fields if field.name == "confirmation_spread"
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

        failure_stamp = Time(nanoseconds=base_ns + 600_000_000).to_msg()
        publishers["odom"].publish(
            _odom(Time(nanoseconds=base_ns + 650_000_000).to_msg(), 0.1)
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

        recovery_stamp = Time(nanoseconds=base_ns + 700_000_000).to_msg()
        publishers["odom"].publish(
            _odom(Time(nanoseconds=base_ns + 750_000_000).to_msg(), 0.1)
        )
        executor.spin_once(timeout_sec=0.02)
        publishers["depth"].publish(_depth(recovery_stamp, object_center_column=302))
        _spin_until(
            executor,
            lambda: _values(health).get("geometry.degradation_reason")
            == "recovery:reconfirmation_required",
        )
        assert len(clouds) == 1

        reconfirmation_stamp = Time(nanoseconds=base_ns + 800_000_000).to_msg()
        publishers["odom"].publish(
            _odom(Time(nanoseconds=base_ns + 850_000_000).to_msg(), 0.1)
        )
        executor.spin_once(timeout_sec=0.02)
        publishers["depth"].publish(
            _depth(reconfirmation_stamp, object_center_column=302)
        )
        _spin_until(
            executor,
            lambda: len(clouds) == 2
            and _values(health).get("geometry.degradation_reason") == "",
        )
        recovered_points = [
            struct.unpack_from("<fff", bytes(clouds[-1].data), offset)
            for offset in range(0, len(clouds[-1].data), clouds[-1].point_step)
        ]
        recovered_centroid_x = sum(point[0] for point in recovered_points) / len(
            recovered_points
        )
        assert abs(recovered_centroid_x - centroid_x) < 0.025
        values = _values(health)
        assert values["geometry.active_retained_hazard_count"] == "1"
        assert values["geometry.degradation_reason"] == ""

        _spin_until(executor, lambda: len(clouds) == 3, timeout=4.0)
        clearing = clouds[-1]
        assert clearing.width == 0
        clearing_stamp_ns = (
            clearing.header.stamp.sec * 1_000_000_000 + clearing.header.stamp.nanosec
        )
        reconfirmation_stamp_ns = (
            reconfirmation_stamp.sec * 1_000_000_000 + reconfirmation_stamp.nanosec
        )
        assert clearing_stamp_ns - reconfirmation_stamp_ns >= 2_000_000_000
    finally:
        executor.remove_node(driver)
        executor.remove_node(perception)
        driver.destroy_node()
        perception.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
