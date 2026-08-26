"""Deterministic black-box replay for confirmed low-profile hazards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import subprocess
import time
from collections.abc import Callable
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2

from .cable_replay_audit import NegativeRegion, audit_rgb_cable_replay
from .geometric_replay_audit import (
    GeometricReplayCloud,
    observation_blind_zone_retention_audit,
)
from .replay_result import ReplayResultAccumulator
from .temporal import EvidenceMask, OdomPoseCache, Pose3


def _stamp_ns(message: object) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


class _Collector(Node):
    def __init__(self, health_topic: str, cloud_topic: str) -> None:
        super().__init__("geometric_hazard_replay_collector")
        self.latest_health: DiagnosticArray | None = None
        self._health: ReplayResultAccumulator | None = None
        self.clouds: list[dict[str, Any]] = []
        self.maximum_observed_speed_mps = 0.0
        self.maximum_pending_work = 0
        self.maximum_processing_p95_ms: float | None = None
        self.maximum_depth_cpu_cores: float | None = None
        self.maximum_memory_growth_bytes: int | None = None
        self._first_odom_stamp_ns: int | None = None
        self._last_odom_stamp_ns: int | None = None
        self._odom = self._new_odom_cache()
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
        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            DiagnosticArray, health_topic, self._collect_health, health_qos
        )
        self.create_subscription(
            PointCloud2, cloud_topic, self._collect_cloud, cloud_qos
        )
        self.create_subscription(Odometry, "/odom", self._collect_odom, odom_qos)

    @staticmethod
    def _new_odom_cache() -> OdomPoseCache:
        return OdomPoseCache(
            maximum_samples=100_000,
            maximum_age_ns=3_600_000_000_000,
        )

    def _collect_health(self, message: DiagnosticArray) -> None:
        if not message.status:
            return
        self.latest_health = message
        status = message.status[0]
        if self._health is None:
            self._health = ReplayResultAccumulator(diagnostic_name=status.name)
        values = {item.key: item.value for item in status.values}
        self._health.record(
            state=status.message,
            values=values,
        )
        pending = [
            int(float(value))
            for key, value in values.items()
            if key.endswith("pending_count") and value not in ("", "unknown")
        ]
        self.maximum_pending_work = max(
            self.maximum_pending_work, max(pending, default=0)
        )
        self.maximum_processing_p95_ms = _maximum_optional(
            self.maximum_processing_p95_ms,
            values.get("stage.perception.processing_wall_p95_ms"),
        )
        self.maximum_depth_cpu_cores = _maximum_optional(
            self.maximum_depth_cpu_cores,
            values.get("stage.depth_geometry.average_cpu_cores"),
        )
        memory_growth = _maximum_optional(
            (
                float(self.maximum_memory_growth_bytes)
                if self.maximum_memory_growth_bytes is not None
                else None
            ),
            values.get("resource.memory_growth_bytes"),
        )
        self.maximum_memory_growth_bytes = (
            max(0, int(memory_growth)) if memory_growth is not None else None
        )

    def _collect_odom(self, message: Odometry) -> None:
        stamp_ns = _stamp_ns(message.header.stamp)
        if stamp_ns <= 0:
            return
        pose = message.pose.pose
        self._odom.add(
            stamp_ns,
            Pose3(
                translation=(
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ),
                rotation=(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
            ),
        )
        self._first_odom_stamp_ns = (
            stamp_ns
            if self._first_odom_stamp_ns is None
            else min(self._first_odom_stamp_ns, stamp_ns)
        )
        self._last_odom_stamp_ns = (
            stamp_ns
            if self._last_odom_stamp_ns is None
            else max(self._last_odom_stamp_ns, stamp_ns)
        )
        twist = message.twist.twist.linear
        self.maximum_observed_speed_mps = max(
            self.maximum_observed_speed_mps,
            math.hypot(float(twist.x), float(twist.y)),
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
        evidence_field = next(
            (
                field
                for field in message.fields
                if field.name == "evidence_mask"
            ),
            None,
        )
        latency_field = next(
            (
                field
                for field in message.fields
                if field.name == "confirmation_latency_ms"
            ),
            None,
        )
        group_field = next(
            (
                field
                for field in message.fields
                if field.name == "cloud_group_index"
            ),
            None,
        )
        track_field = next(
            (
                field
                for field in message.fields
                if field.name == "hazard_track_id"
            ),
            None,
        )
        offsets = list(
            range(0, len(message.data), int(message.point_step))
        )
        points = [
            struct.unpack_from("<fff", bytes(message.data), offset)
            for offset in offsets
        ]
        evidence_masks = (
            [
                struct.unpack_from(
                    "<B",
                    bytes(message.data),
                    offset + evidence_field.offset,
                )[0]
                for offset in offsets
            ]
            if evidence_field is not None
            else [0] * len(points)
        )
        confirmation_spreads = (
            [
                struct.unpack_from(
                    "<f", bytes(message.data), offset + spread_field.offset
                )[0]
                for offset in offsets
            ]
            if spread_field is not None
            else [None] * len(points)
        )
        confirmation_latencies_ms = (
            [
                struct.unpack_from(
                    "<f", bytes(message.data), offset + latency_field.offset
                )[0]
                for offset in offsets
            ]
            if latency_field is not None
            else [None] * len(points)
        )
        group_ids = (
            [
                struct.unpack_from(
                    "<I", bytes(message.data), offset + group_field.offset
                )[0]
                for offset in offsets
            ]
            if group_field is not None
            else [0] * len(points)
        )
        track_ids = (
            [
                struct.unpack_from(
                    "<I", bytes(message.data), offset + track_field.offset
                )[0]
                for offset in offsets
            ]
            if track_field is not None
            else group_ids
        )
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
                for offset in offsets
            ]
        metrics = _point_metrics(
            points,
            evidence_masks,
            source_stamps_ns,
            confirmation_spreads,
            confirmation_latencies_ms,
        )
        hazard_groups = []
        for group_index in sorted(set(group_ids)):
            indices = [
                index
                for index, value in enumerate(group_ids)
                if value == group_index
            ]
            hazard_groups.append(
                {
                    "cloud_group_index": group_index,
                    "hazard_track_id": track_ids[indices[0]],
                    **_point_metrics(
                        [points[index] for index in indices],
                        [evidence_masks[index] for index in indices],
                        (
                            [source_stamps_ns[index] for index in indices]
                            if source_stamps_ns
                            else []
                        ),
                        [confirmation_spreads[index] for index in indices],
                        [
                            confirmation_latencies_ms[index]
                            for index in indices
                        ],
                    ),
                }
            )
        self.clouds.append(
            {
                "stamp_ns": _stamp_ns(message.header.stamp),
                "frame_id": message.header.frame_id,
                "point_count": int(message.width) * int(message.height),
                "point_step": int(message.point_step),
                "data_sha256": hashlib.sha256(bytes(message.data)).hexdigest(),
                **metrics,
                "hazard_groups": hazard_groups,
                "clearing": not points,
            }
        )

    def clear(self) -> None:
        self.latest_health = None
        self._health = None
        self.clouds = []
        self.maximum_observed_speed_mps = 0.0
        self.maximum_pending_work = 0
        self.maximum_processing_p95_ms = None
        self.maximum_depth_cpu_cores = None
        self.maximum_memory_growth_bytes = None
        self._first_odom_stamp_ns = None
        self._last_odom_stamp_ns = None
        self._odom = self._new_odom_cache()

    def report(self) -> dict[str, Any]:
        if self._health is None:
            raise RuntimeError("replay produced no perception-health output")
        health = self._health.report()
        clouds = [self._with_detection_distance(cloud) for cloud in self.clouds]
        duration_seconds = None
        if (
            self._first_odom_stamp_ns is not None
            and self._last_odom_stamp_ns is not None
        ):
            duration_seconds = (
                self._last_odom_stamp_ns - self._first_odom_stamp_ns
            ) / 1_000_000_000
        return {
            "health": health,
            "clouds": clouds,
            "runtime": {
                "evaluated_duration_seconds": duration_seconds,
                "observed_maximum_speed_mps": round(
                    self.maximum_observed_speed_mps, 6
                ),
                "resource_use": {
                    "processing_p95_ms": self.maximum_processing_p95_ms,
                    "depth_geometry_average_cpu_cores": (
                        self.maximum_depth_cpu_cores
                    ),
                    "memory_growth_bytes": self.maximum_memory_growth_bytes,
                    "maximum_pending_work": self.maximum_pending_work,
                },
            },
        }

    def _with_detection_distance(self, cloud: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(cloud)
        enriched["hazard_groups"] = [
            self._with_detection_distance(group)
            for group in cloud.get("hazard_groups", [])
        ]
        stamp_ns = cloud.get("source_stamp_max_ns")
        pose = self._odom.interpolate(int(stamp_ns)) if stamp_ns else None
        if pose is None:
            enriched["confirmed_detection_distance_m"] = None
            enriched["rgb_cable_confirmed_detection_distance_m"] = None
            return enriched
        robot_x, robot_y, _ = pose.translation
        for prefix in ("", "rgb_cable_"):
            x_value = cloud.get(f"{prefix}centroid_x_m")
            y_value = cloud.get(f"{prefix}centroid_y_m")
            key = f"{prefix}confirmed_detection_distance_m"
            enriched[key] = (
                round(
                    math.hypot(
                        float(x_value) - robot_x,
                        float(y_value) - robot_y,
                    ),
                    6,
                )
                if x_value is not None and y_value is not None
                else None
            )
        return enriched


def _point_metrics(
    points: list[tuple[float, float, float]],
    evidence_masks: list[int],
    source_stamps_ns: list[int],
    confirmation_spreads: list[float | None],
    confirmation_latencies_ms: list[float | None],
) -> dict[str, object]:
    rgb_cable_points = [
        point
        for point, evidence_mask in zip(points, evidence_masks, strict=True)
        if evidence_mask & int(EvidenceMask.RGB_CABLE)
    ]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    z_values = sorted(point[2] for point in points)
    spreads = [value for value in confirmation_spreads if value is not None]
    latencies = [
        value for value in confirmation_latencies_ms if value is not None
    ]
    return {
        "point_count": len(points),
        "p20_height_m": _percentile(z_values, 0.20),
        "p90_height_m": _percentile(z_values, 0.90),
        "horizontal_span_m": round(_horizontal_span(points), 6),
        "centroid_x_m": (
            round(sum(x_values) / len(x_values), 6) if x_values else None
        ),
        "centroid_y_m": (
            round(sum(y_values) / len(y_values), 6) if y_values else None
        ),
        "confirmation_spread_m": (
            round(max(spreads), 6) if spreads else None
        ),
        "rgb_cable_point_count": len(rgb_cable_points),
        "rgb_cable_span_m": round(_horizontal_span(rgb_cable_points), 6),
        "rgb_cable_centroid_x_m": (
            round(
                sum(point[0] for point in rgb_cable_points)
                / len(rgb_cable_points),
                6,
            )
            if rgb_cable_points
            else None
        ),
        "rgb_cable_centroid_y_m": (
            round(
                sum(point[1] for point in rgb_cable_points)
                / len(rgb_cable_points),
                6,
            )
            if rgb_cable_points
            else None
        ),
        "source_stamp_min_ns": (
            min(source_stamps_ns) if source_stamps_ns else None
        ),
        "source_stamp_max_ns": (
            max(source_stamps_ns) if source_stamps_ns else None
        ),
        "confirmation_latency_ms": (
            round(max(latencies), 6) if latencies else None
        ),
        "evidence_mask": _combined_evidence_mask(evidence_masks),
    }


def _combined_evidence_mask(evidence_masks: list[int]) -> int:
    combined = 0
    for evidence_mask in evidence_masks:
        combined |= evidence_mask
    return combined


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
    playback_topics: tuple[str, ...] = (),
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
        playback_command = [
                "ros2",
                "bag",
                "play",
                str(bag),
                "--clock",
                "--rate",
                str(rate),
                "--read-ahead-queue-size",
                "1000",
            ]
        if playback_topics:
            playback_command.extend(["--topics", *playback_topics])
        playback = subprocess.Popen(
            playback_command,
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


def _parse_args(
    args: list[str] | None,
    *,
    evidence_mode: str,
) -> argparse.Namespace:
    target = (
        "reference strong protrusion"
        if evidence_mode == "geometry"
        else "training-free RGB cable"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Replay independent RGB-D/odom inputs and require a deterministic "
            f"confirmed odom point cloud from the {target}."
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
        "--expected-cable-center",
        type=float,
        nargs=2,
        metavar=("ODOM_X", "ODOM_Y"),
    )
    parser.add_argument("--expected-cable-radius", type=float, default=0.15)
    parser.add_argument(
        "--minimum-cable-physical-span", type=float, default=0.06
    )
    parser.add_argument(
        "--negative-cable-region",
        nargs=5,
        action="append",
        metavar=("LABEL", "MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help=(
            "Reject persistent RGB-cable evidence in a named annotated odom "
            "region; use labels from the negative replay manifest."
        ),
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Audit a cable-negative scene without requiring a positive event.",
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
    if parsed.minimum_cable_physical_span <= 0.0:
        parser.error("--minimum-cable-physical-span must be positive")
    try:
        parsed.negative_cable_region = tuple(
            NegativeRegion(
                label=values[0],
                minimum_x=float(values[1]),
                maximum_x=float(values[2]),
                minimum_y=float(values[3]),
                maximum_y=float(values[4]),
            )
            for values in parsed.negative_cable_region or ()
        )
    except ValueError:
        parser.error("negative cable region bounds must be numeric")
    if parsed.negative_only and not parsed.negative_cable_region:
        parser.error("--negative-only requires --negative-cable-region")
    if not parsed.bag.exists():
        parser.error(f"bag does not exist: {parsed.bag}")
    return parsed


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = int(round((len(values) - 1) * fraction))
    return round(values[index], 6)


def _horizontal_span(points: list[tuple[float, float, float]]) -> float:
    if not points:
        return 0.0
    return (
        (max(point[0] for point in points) - min(point[0] for point in points))
        ** 2
        + (max(point[1] for point in points) - min(point[1] for point in points))
        ** 2
    ) ** 0.5


def _maximum_optional(current: float | None, raw: str | None) -> float | None:
    if raw in (None, "", "unknown"):
        return current
    try:
        value = float(raw)
    except ValueError:
        return current
    return value if current is None else max(current, value)


def _canonical(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "health": result["health"]["canonical"],
        "clouds": result["clouds"],
        "runtime": {
            "evaluated_duration_seconds": result["runtime"][
                "evaluated_duration_seconds"
            ],
            "observed_maximum_speed_mps": result["runtime"][
                "observed_maximum_speed_mps"
            ],
        },
    }


def main(args: list[str] | None = None) -> None:
    _main(args, evidence_mode="geometry")


def main_rgb_cable(args: list[str] | None = None) -> None:
    _main(args, evidence_mode="rgb_cable")


def _main(args: list[str] | None, *, evidence_mode: str) -> None:
    options = _parse_args(args, evidence_mode=evidence_mode)
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
                f"non-deterministic {evidence_mode} replay on run "
                f"{run_number}:\n"
                + json.dumps(
                    {"first": baseline, "different": _canonical(result)},
                    indent=2,
                    sort_keys=True,
                )
            )
    clouds = baseline["clouds"]
    hazard_clouds = [cloud for cloud in clouds if not cloud["clearing"]]
    negative_only = evidence_mode == "rgb_cable" and options.negative_only
    if not hazard_clouds and not negative_only:
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
    if evidence_mode == "geometry" and not strong_clouds:
        raise SystemExit(
            "reference replay produced no robustly supported strong "
            "protrusion cloud"
        )
    if evidence_mode == "geometry" and any(
        cloud["horizontal_span_m"] > 0.75 for cloud in hazard_clouds
    ):
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
    if (
        evidence_mode == "geometry"
        and options.expected_power_strip_center is not None
    ):
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
    cable_audit = None
    if evidence_mode == "rgb_cable":
        try:
            cable_audit = audit_rgb_cable_replay(
                stable_values=values,
                clouds=clouds,
                maximum_alignment_spread_m=(
                    options.maximum_alignment_spread
                ),
                minimum_physical_span_m=(
                    options.minimum_cable_physical_span
                ),
                expected_center_odom=(
                    tuple(options.expected_cable_center)
                    if options.expected_cable_center is not None
                    else None
                ),
                expected_center_radius_m=options.expected_cable_radius,
                negative_regions=options.negative_cable_region,
                require_positive=not negative_only,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"RGB cable replay audit failed: {error}") from error
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
    blind_zone_audit = None
    if not negative_only:
        confirmation_spread = float(
            values["geometry.latest_confirmation_spread_m"]
        )
        if confirmation_spread > options.maximum_alignment_spread:
            raise SystemExit(
                "confirmed observations exceeded the replay alignment spread: "
                f"{confirmation_spread:.3f} m"
            )
        try:
            audit_clouds = [
                GeometricReplayCloud(
                    clearing=bool(cloud["clearing"]),
                    source_stamp_max_ns=cloud["source_stamp_max_ns"],
                    stamp_ns=int(cloud["stamp_ns"]),
                )
                for cloud in clouds
            ]
            blind_zone_audit = observation_blind_zone_retention_audit(
                audit_clouds,
                latest_processed_depth_stamp_ns=int(
                    values["geometry.latest_processed_depth_stamp_ns"]
                ),
                minimum_retention_ns=int(
                    float(values["geometry.confirmed_retention_ms"])
                    * 1_000_000
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                f"blind-zone retention audit failed: {error}"
            ) from error
    output = json.dumps(
        {
            "repeat_count": options.repeat,
            "measured_camera_height_m": measured_height,
            "confirmed_cloud_count": len(hazard_clouds),
            "clearing_cloud_count": len(clouds) - len(hazard_clouds),
            "observation_blind_zone_retention": blind_zone_audit,
            "rgb_cable_audit": cable_audit,
            "runtime": baseline["runtime"],
            "canonical": baseline,
            "runs": results,
        },
        indent=2,
        sort_keys=True,
    )
    if options.output:
        options.output.write_text(output + "\n", encoding="utf-8")
    print(output)
