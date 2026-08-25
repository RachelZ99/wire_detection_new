"""Deterministic black-box replay for the confirmed geometric hazard seam."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
from collections.abc import Callable
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2

from .replay_result import ReplayResultAccumulator


def _stamp_ns(message: object) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


class _Collector(Node):
    def __init__(self, health_topic: str, cloud_topic: str) -> None:
        super().__init__("geometric_hazard_replay_collector")
        self.latest_health: DiagnosticArray | None = None
        self._health: ReplayResultAccumulator | None = None
        self.clouds: list[dict[str, Any]] = []
        health_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            DiagnosticArray, health_topic, self._collect_health, health_qos
        )
        self.create_subscription(
            PointCloud2, cloud_topic, self._collect_cloud, cloud_qos
        )

    def _collect_health(self, message: DiagnosticArray) -> None:
        if not message.status:
            return
        self.latest_health = message
        status = message.status[0]
        if self._health is None:
            self._health = ReplayResultAccumulator(diagnostic_name=status.name)
        self._health.record(
            state=status.message,
            values={item.key: item.value for item in status.values},
        )

    def _collect_cloud(self, message: PointCloud2) -> None:
        self.clouds.append(
            {
                "stamp_ns": _stamp_ns(message.header.stamp),
                "frame_id": message.header.frame_id,
                "point_count": int(message.width) * int(message.height),
                "point_step": int(message.point_step),
                "data_sha256": hashlib.sha256(bytes(message.data)).hexdigest(),
            }
        )

    def clear(self) -> None:
        self.latest_health = None
        self._health = None
        self.clouds = []

    def report(self) -> dict[str, Any]:
        if self._health is None:
            raise RuntimeError("replay produced no perception-health output")
        return {"health": self._health.report(), "clouds": self.clouds}


def _spin_until(
    collector: _Collector,
    predicate: Callable[[], bool],
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(collector, timeout_sec=0.1)
        if predicate():
            return True
    return False


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


def _run_once(
    bag: Path,
    *,
    health_topic: str,
    cloud_topic: str,
    rate: float,
    startup_timeout: float,
) -> dict[str, Any]:
    collector = _Collector(health_topic, cloud_topic)
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "low_profile_hazard_perception",
            "geometric_hazard.launch.py",
            "use_sim_time:=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _spin_until(
            collector,
            lambda: collector.latest_health is not None,
            startup_timeout,
        ):
            raise RuntimeError(
                "geometric hazard node did not become observable"
            )
        collector.clear()
        playback = subprocess.Popen(
            [
                "ros2",
                "bag",
                "play",
                str(bag),
                "--clock",
                "--rate",
                str(rate),
                "--read-ahead-queue-size",
                "1000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        while playback.poll() is None:
            rclpy.spin_once(collector, timeout_sec=0.1)
        stderr = playback.communicate()[1].decode("utf-8", errors="replace")
        if playback.returncode != 0:
            raise RuntimeError(f"rosbag replay failed: {stderr.strip()}")
        _spin_until(collector, lambda: False, 1.0)
        return collector.report()
    finally:
        _stop(launch)
        collector.destroy_node()


def _parse_args(args: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay independent RGB-D/odom inputs and require a deterministic "
            "confirmed odom point cloud from the reference strong protrusion."
        )
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--minimum-measured-height", type=float, default=0.20)
    parser.add_argument("--maximum-measured-height", type=float, default=0.25)
    parser.add_argument(
        "--health-topic", default="/low_profile_hazard_perception/health"
    )
    parser.add_argument(
        "--cloud-topic",
        default="/low_profile_hazard_perception/confirmed_hazards",
    )
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(args)
    if parsed.repeat < 2:
        parser.error("--repeat must be at least 2")
    if parsed.rate <= 0.0:
        parser.error("--rate must be positive")
    if not parsed.bag.exists():
        parser.error(f"bag does not exist: {parsed.bag}")
    return parsed


def _canonical(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "health": result["health"]["canonical"],
        "clouds": result["clouds"],
    }


def main(args: list[str] | None = None) -> None:
    options = _parse_args(args)
    rclpy.init(args=[])
    try:
        results = [
            _run_once(
                options.bag,
                health_topic=options.health_topic,
                cloud_topic=options.cloud_topic,
                rate=options.rate,
                startup_timeout=options.startup_timeout,
            )
            for _ in range(options.repeat)
        ]
    finally:
        rclpy.shutdown()
    baseline = _canonical(results[0])
    for run_number, result in enumerate(results[1:], start=2):
        if _canonical(result) != baseline:
            raise SystemExit(
                f"non-deterministic geometric replay on run {run_number}:\n"
                + json.dumps(
                    {"first": baseline, "different": _canonical(result)},
                    indent=2,
                    sort_keys=True,
                )
            )
    clouds = baseline["clouds"]
    if not clouds:
        raise SystemExit("reference replay produced no confirmed hazard cloud")
    if any(
        cloud["frame_id"] != "odom"
        or cloud["stamp_ns"] <= 0
        or cloud["point_count"] <= 0
        for cloud in clouds
    ):
        raise SystemExit(
            "operational clouds must be non-empty, stamped in odom"
        )
    values = baseline["health"]["stable_values"]
    measured_height = float(values["ground.camera_height_m"])
    if not (
        options.minimum_measured_height
        <= measured_height
        <= options.maximum_measured_height
    ):
        raise SystemExit(
            "observed ground height is outside the measured installation "
            f"range: {measured_height:.3f} m"
        )
    output = json.dumps(
        {
            "repeat_count": options.repeat,
            "measured_camera_height_m": measured_height,
            "confirmed_cloud_count": len(clouds),
            "canonical": baseline,
            "runs": results,
        },
        indent=2,
        sort_keys=True,
    )
    if options.output:
        options.output.write_text(output + "\n", encoding="utf-8")
    print(output)
