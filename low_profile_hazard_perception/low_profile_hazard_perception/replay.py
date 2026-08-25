"""Run the highest test seam against a rosbag and compare repeated results."""

from __future__ import annotations

import argparse
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


class _HealthCollector(Node):
    def __init__(self, health_topic: str) -> None:
        super().__init__("replay_health_collector")
        self.latest: DiagnosticArray | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscription = self.create_subscription(
            DiagnosticArray, health_topic, self._collect, qos
        )

    def _collect(self, message: DiagnosticArray) -> None:
        if message.status:
            self.latest = message


def _spin_until(
    collector: _HealthCollector,
    predicate: Callable[[], bool],
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(collector, timeout_sec=0.1)
        if predicate():
            return True
    return False


def _canonical(message: DiagnosticArray) -> dict[str, Any]:
    status = message.status[0]
    values = {item.key: item.value for item in status.values}
    volatile_suffixes = (".sensor_stamp_age_ms", ".receive_age_ms")
    stable_values = {
        key: value
        for key, value in sorted(values.items())
        if not key.endswith(volatile_suffixes)
    }
    age_fields = sorted(
        key for key in values if key.endswith(volatile_suffixes)
    )
    return {
        "diagnostic_name": status.name,
        "state": status.message,
        "stable_values": stable_values,
        "age_fields_present": age_fields,
    }


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
    health_topic: str,
    rate: float,
    startup_timeout: float,
) -> dict[str, Any]:
    collector = _HealthCollector(health_topic)
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "low_profile_hazard_perception",
            "input_health.launch.py",
            "use_sim_time:=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ready = _spin_until(
            collector,
            lambda: collector.latest is not None,
            startup_timeout,
        )
        if not ready:
            raise RuntimeError("input health node did not become observable")
        collector.latest = None
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
        if collector.latest is None:
            raise RuntimeError("replay produced no perception-health result")
        return _canonical(collector.latest)
    finally:
        _stop(launch)
        collector.destroy_node()


def _parse_args(args: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay RGB, depth, CameraInfo, TF and odom and require identical "
            "canonical health output on every run."
        )
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument(
        "--health-topic",
        default="/low_profile_hazard_perception/health",
    )
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(args)
    if parsed.repeat < 2:
        parser.error("--repeat must be at least 2")
    if parsed.rate <= 0.0:
        parser.error("--rate must be positive")
    if not parsed.bag.exists():
        parser.error(f"bag does not exist: {parsed.bag}")
    return parsed


def main(args: list[str] | None = None) -> None:
    options = _parse_args(args)
    rclpy.init()
    try:
        results = [
            _run_once(
                options.bag,
                options.health_topic,
                options.rate,
                options.startup_timeout,
            )
            for _ in range(options.repeat)
        ]
    finally:
        rclpy.shutdown()
    baseline = results[0]
    for run_number, result in enumerate(results[1:], start=2):
        if result != baseline:
            raise SystemExit(
                "non-deterministic health result on run "
                f"{run_number}:\n"
                + json.dumps(
                    {"first": baseline, "different": result},
                    indent=2,
                    sort_keys=True,
                )
            )
    if baseline["state"] != "HEALTHY":
        raise SystemExit(
            "replay was deterministic but input health was not HEALTHY:\n"
            + json.dumps(baseline, indent=2, sort_keys=True)
        )
    output = json.dumps(
        {"repeat_count": options.repeat, "health": baseline},
        indent=2,
        sort_keys=True,
    )
    if options.output:
        options.output.write_text(output + "\n", encoding="utf-8")
    print(output)
