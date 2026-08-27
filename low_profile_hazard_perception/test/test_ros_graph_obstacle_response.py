import struct
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("diagnostic_msgs")
pytest.importorskip("sensor_msgs")

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

from low_profile_hazard_perception.obstacle_response import (  # noqa: E402
    ObstacleResponseConfig,
    RecordingObstacleResponsePort,
)
from low_profile_hazard_perception.obstacle_response_ros import (  # noqa: E402
    ObstacleResponseAdapterNode,
)


def _spin_until(executor, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return
    raise AssertionError("ROS graph condition did not become true")


def _health(stamp_ns):
    message = DiagnosticArray()
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    status = DiagnosticStatus()
    status.name = "low_profile_hazard_perception/input_health"
    status.message = "HEALTHY"
    values = {
        "state": "HEALTHY",
        "profile.id": "dcw2-home-640x360-v1",
        "profile.binding_state": "BOUND",
        "profile.maximum_speed_mps": "0.3",
        "profile.latest_observed_speed_mps": "0.3",
        "cable.provider": "training_free_thin_line",
        "resource.npu_state": "disabled_rule_profile",
        "candidate.count": "999",
    }
    status.values = [KeyValue(key=key, value=value) for key, value in values.items()]
    message.status = [status]
    return message


def _cloud(stamp_ns):
    message = PointCloud2()
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    message.header.frame_id = "odom"
    message.height = 1
    message.width = 1
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
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
            name="cloud_group_index",
            offset=32,
            datatype=PointField.UINT32,
            count=1,
        ),
        PointField(
            name="hazard_track_id",
            offset=36,
            datatype=PointField.UINT32,
            count=1,
        ),
    ]
    seconds, nanoseconds = divmod(stamp_ns, 1_000_000_000)
    message.point_step = 40
    message.row_step = 40
    message.data = struct.pack(
        "<ffffiIB3xfII",
        1.0,
        0.0,
        0.02,
        0.01,
        seconds,
        nanoseconds,
        1,
        100.0,
        0,
        4,
    )
    message.is_dense = True
    return message


def test_adapter_consumes_only_confirmed_cloud_and_top_level_health():
    rclpy.init(args=[])
    port = RecordingObstacleResponsePort()
    adapter = ObstacleResponseAdapterNode(
        port=port,
        config=ObstacleResponseConfig(
            expected_profile_id="dcw2-home-640x360-v1",
            maximum_speed_mps=0.3,
            health_timeout_ns=600_000_000,
            maximum_observation_age_ns=2_500_000_000,
        ),
    )
    driver = Node("obstacle_response_contract_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    executor.add_node(driver)
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    health_publisher = driver.create_publisher(
        DiagnosticArray,
        "/low_profile_hazard_perception/health",
        qos,
    )
    cloud_publisher = driver.create_publisher(
        PointCloud2,
        "/low_profile_hazard_perception/confirmed_hazards",
        qos,
    )
    try:
        _spin_until(
            executor,
            lambda: health_publisher.get_subscription_count() == 1
            and cloud_publisher.get_subscription_count() == 1,
        )
        now_ns = adapter.get_clock().now().nanoseconds
        cloud_publisher.publish(_cloud(now_ns))
        _spin_until(executor, lambda: bool(port.statuses))
        assert not port.snapshots
        health_publisher.publish(_health(now_ns))
        _spin_until(executor, lambda: len(port.snapshots) == 1)

        assert port.snapshots[0].hazards[0].hazard_track_id == 4
        subscriptions = dict(
            adapter.get_subscriber_names_and_types_by_node(
                adapter.get_name(), adapter.get_namespace()
            )
        )
        assert (
            "/low_profile_hazard_perception/candidate_diagnostics" not in subscriptions
        )
        assert "/cmd_vel" not in subscriptions
        assert all(
            topic != "/cmd_vel"
            for topic, _ in adapter.get_publisher_names_and_types_by_node(
                adapter.get_name(), adapter.get_namespace()
            )
        )
    finally:
        executor.remove_node(driver)
        executor.remove_node(adapter)
        driver.destroy_node()
        adapter.destroy_node()
        rclpy.shutdown()
