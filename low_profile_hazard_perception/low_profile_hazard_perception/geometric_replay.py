"""Deterministic black-box replay for the confirmed geometric hazard seam."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import struct
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
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
        spread_field = next(
            (
                field
                for field in message.fields
                if field.name == "confirmation_spread"
            ),
            None,
        )
        stamp_sec_field = next(
            (
                field
                for field in message.fields
                if field.name == "observation_stamp_sec"
            ),
            None,
        )
        stamp_nanosec_field = next(
            (
                field
                for field in message.fields
                if field.name == "observation_stamp_nanosec"
            ),
            None,
        )
        confirmation_spread_m = None
        if spread_field is not None and message.data:
            confirmation_spread_m = round(
                max(
                    struct.unpack_from(
                        "<f",
                        bytes(message.data),
                        offset + spread_field.offset,
                    )[0]
                    for offset in range(
                        0, len(message.data), int(message.point_step)
                    )
                ),
                6,
            )
        points = [
            struct.unpack_from("<fff", bytes(message.data), offset)
            for offset in range(0, len(message.data), int(message.point_step))
        ]
        source_stamps_ns: list[int] = []
        if (
            stamp_sec_field is not None
            and stamp_nanosec_field is not None
            and message.data
        ):
            source_stamps_ns = [
                struct.unpack_from(
                    "<i",
                    bytes(message.data),
                    offset + stamp_sec_field.offset,
                )[0]
                * 1_000_000_000
                + struct.unpack_from(
                    "<I",
                    bytes(message.data),
                    offset + stamp_nanosec_field.offset,
                )[0]
                for offset in range(
                    0, len(message.data), int(message.point_step)
                )
            ]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        z_values = sorted(point[2] for point in points)
        horizontal_span_m = 0.0
        if points:
            horizontal_span_m = (
                (max(x_values) - min(x_values)) ** 2
                + (max(y_values) - min(y_values)) ** 2
            ) ** 0.5
        self.clouds.append(
            {
                "stamp_ns": _stamp_ns(message.header.stamp),
                "frame_id": message.header.frame_id,
                "point_count": int(message.width) * int(message.height),
                "point_step": int(message.point_step),
                "data_sha256": hashlib.sha256(bytes(message.data)).hexdigest(),
                "p20_height_m": _percentile(z_values, 0.20),
                "p90_height_m": _percentile(z_values, 0.90),
                "horizontal_span_m": round(horizontal_span_m, 6),
                "centroid_x_m": (
                    round(sum(x_values) / len(x_values), 6)
                    if x_values
                    else None
                ),
                "centroid_y_m": (
                    round(sum(y_values) / len(y_values), 6)
                    if y_values
                    else None
                ),
                "confirmation_spread_m": confirmation_spread_m,
                "source_stamp_min_ns": (
                    min(source_stamps_ns) if source_stamps_ns else None
                ),
                "source_stamp_max_ns": (
                    max(source_stamps_ns) if source_stamps_ns else None
                ),
                "clearing": not points,
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
        "--maximum-alignment-spread", type=float, default=0.025
    )
    parser.add_argument(
        "--expected-power-strip-center",
        type=float,
        nargs=2,
        metavar=("ODOM_X", "ODOM_Y"),
    )
    parser.add_argument(
        "--expected-power-strip-radius", type=float, default=0.15
    )
    parser.add_argument(
        "--reflective-floor-region",
        type=float,
        nargs=4,
        action="append",
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help=(
            "Reject a persistent (two-cloud) confirmed event in this "
            "annotated odom region; may be supplied more than once."
        ),
    )
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
    if parsed.maximum_alignment_spread <= 0.0:
        parser.error("--maximum-alignment-spread must be positive")
    if not parsed.bag.exists():
        parser.error(f"bag does not exist: {parsed.bag}")
    return parsed


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = int(round((len(values) - 1) * fraction))
    return round(values[index], 6)


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
    hazard_clouds = [cloud for cloud in clouds if not cloud["clearing"]]
    if not hazard_clouds:
        raise SystemExit("reference replay produced no confirmed hazard cloud")
    if any(
        cloud["frame_id"] != "odom"
        or cloud["stamp_ns"] <= 0
        for cloud in clouds
    ):
        raise SystemExit("operational clouds must be stamped in odom")
    if any(
        cloud["source_stamp_min_ns"] is None
        or cloud["source_stamp_min_ns"] <= 0
        or cloud["stamp_ns"] != cloud["source_stamp_min_ns"]
        for cloud in hazard_clouds
    ):
        raise SystemExit(
            "hazard points must retain source stamps and the cloud header "
            "must conservatively use the oldest source stamp"
        )
    strong_clouds = [
        cloud
        for cloud in hazard_clouds
        if cloud["p20_height_m"] is not None
        and cloud["p20_height_m"] >= 0.014
        and cloud["p90_height_m"] <= 0.151
        and cloud["horizontal_span_m"] >= 0.04
    ]
    if not strong_clouds:
        raise SystemExit(
            "reference replay produced no robustly supported strong "
            "protrusion cloud"
        )
    if any(cloud["horizontal_span_m"] > 0.75 for cloud in hazard_clouds):
        raise SystemExit(
            "an operational cloud has a trail-like spatial extent"
        )
    if any(
        cloud["confirmation_spread_m"] is None
        or cloud["confirmation_spread_m"] > options.maximum_alignment_spread
        for cloud in hazard_clouds
    ):
        raise SystemExit(
            "at least one confirmed cloud exceeded the replay alignment "
            "spread"
        )
    if options.expected_power_strip_center is not None:
        expected_x, expected_y = options.expected_power_strip_center
        matching_power_strip = [
            cloud
            for cloud in strong_clouds
            if (
                (cloud["centroid_x_m"] - expected_x) ** 2
                + (cloud["centroid_y_m"] - expected_y) ** 2
            )
            ** 0.5
            <= options.expected_power_strip_radius
        ]
        if not matching_power_strip:
            raise SystemExit(
                "no confirmed strong protrusion overlaps the annotated "
                "reference power-strip region"
            )
    for region in options.reflective_floor_region or []:
        minimum_x, maximum_x, minimum_y, maximum_y = region
        persistent_count = sum(
            minimum_x <= cloud["centroid_x_m"] <= maximum_x
            and minimum_y <= cloud["centroid_y_m"] <= maximum_y
            for cloud in hazard_clouds
        )
        if persistent_count >= 2:
            raise SystemExit(
                "a persistent confirmed event appeared in an annotated "
                "reflective-floor region"
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
    confirmation_spread = float(
        values["geometry.latest_confirmation_spread_m"]
    )
    if confirmation_spread > options.maximum_alignment_spread:
        raise SystemExit(
            "confirmed observations exceeded the replay alignment spread: "
            f"{confirmation_spread:.3f} m"
        )
    output = json.dumps(
        {
            "repeat_count": options.repeat,
            "measured_camera_height_m": measured_height,
            "confirmed_cloud_count": len(hazard_clouds),
            "clearing_cloud_count": len(clouds) - len(hazard_clouds),
            "canonical": baseline,
            "runs": results,
        },
        indent=2,
        sort_keys=True,
    )
    if options.output:
        options.output.write_text(output + "\n", encoding="utf-8")
    print(output)
