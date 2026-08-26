"""Run a bounded replay soak and emit an auditable profile-budget report."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from .detection_profile import DetectionProfile, default_detection_profile_path
from .resource_budget import (
    BudgetLimits,
    ResourceBudgetAudit,
    ResourceSample,
)


class _SoakCollector(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("detection_profile_soak_collector")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(DiagnosticArray, topic, self._record, qos)
        self.observed = False
        self.processing_samples_ms: list[float] = []
        self.resource_samples: list[ResourceSample] = []
        self.maximum_pending_work = 0
        self.maximum_rgb_reorder_pending = 0
        self.rgb_reorder_capacity = 0
        self.frame_drops = 0
        self.depth_average_cpu_cores: float | None = None
        self.npu_state = "unknown"
        self.profile_id = ""
        self.profile_fingerprint = ""
        self.profile_mismatch_observed = False
        self._last_stage_count = -1
        self._last_resource_elapsed = -1.0
        self.last_health_monotonic = time.monotonic()
        self.last_stage_progress_monotonic: float | None = None

    def _record(self, message: DiagnosticArray) -> None:
        statuses = [
            status
            for status in message.status
            if status.name == "low_profile_hazard_perception/input_health"
        ]
        if not statuses:
            return
        values = {item.key: item.value for item in statuses[0].values}
        self.observed = True
        self.last_health_monotonic = time.monotonic()
        self.profile_id = values.get("profile.id", self.profile_id)
        self.profile_fingerprint = values.get(
            "profile.fingerprint", self.profile_fingerprint
        )
        self.profile_mismatch_observed = self.profile_mismatch_observed or (
            values.get("profile.binding_state") == "MISMATCH"
        )
        stage_count = _integer(values.get("stage.perception.sample_count"))
        if stage_count is not None and stage_count != self._last_stage_count:
            latency = _number(
                values.get("stage.perception.processing_wall_latest_ms")
            )
            if latency is not None:
                self.processing_samples_ms.append(latency)
                self.last_stage_progress_monotonic = time.monotonic()
            self._last_stage_count = stage_count
        depth_cpu = _number(
            values.get("stage.depth_geometry.average_cpu_cores")
        )
        if depth_cpu is not None:
            self.depth_average_cpu_cores = depth_cpu
        resource_elapsed = _number(values.get("resource.elapsed_seconds"))
        rss = _integer(values.get("resource.memory_rss_bytes"))
        process_cpu = _number(values.get("resource.cpu_cores"))
        if (
            resource_elapsed is not None
            and rss is not None
            and process_cpu is not None
            and resource_elapsed != self._last_resource_elapsed
        ):
            self.resource_samples.append(
                ResourceSample(resource_elapsed, rss, process_cpu)
            )
            self._last_resource_elapsed = resource_elapsed
        self.npu_state = values.get("resource.npu_state", self.npu_state)
        pending = [
            _integer(value) or 0
            for key, value in values.items()
            if key.endswith(".pending_count")
        ]
        self.maximum_pending_work = max(
            self.maximum_pending_work, max(pending, default=0)
        )
        self.maximum_rgb_reorder_pending = max(
            self.maximum_rgb_reorder_pending,
            _integer(values.get("queue.pending_rgb_reorder_count")) or 0,
        )
        self.rgb_reorder_capacity = max(
            self.rgb_reorder_capacity,
            _integer(values.get("queue.rgb_reorder_capacity")) or 0,
        )
        drop_keys = (
            key
            for key in values
            if key.endswith(".queue_drops")
            or key in (
                "geometry.pending_depth_drops",
                "cable.pending_rgb_drops",
            )
        )
        self.frame_drops = max(
            self.frame_drops,
            sum(_integer(values.get(key)) or 0 for key in drop_keys),
        )


def _number(value: str | None) -> float | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _integer(value: str | None) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


def _parse_args(args: list[str] | None) -> argparse.Namespace:
    profile = DetectionProfile.load(default_detection_profile_path())
    parser = argparse.ArgumentParser(
        description=(
            "Loop an RGB-D bag and audit the bound detection profile's "
            "latency, CPU, memory, drops, and queue limits."
        )
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float(profile.resource_budget.soak_duration_seconds),
    )
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument(
        "--health-topic", default="/low_profile_hazard_perception/health"
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(args)
    if not parsed.bag.exists():
        parser.error(f"bag does not exist: {parsed.bag}")
    if parsed.duration_seconds <= 0.0 or parsed.rate <= 0.0:
        parser.error("duration and rate must be positive")
    return parsed


def main(args: list[str] | None = None) -> None:
    options = _parse_args(args)
    profile = DetectionProfile.load(default_detection_profile_path())
    collector: _SoakCollector | None = None
    launch: subprocess.Popen[bytes] | None = None
    playback: subprocess.Popen[bytes] | None = None
    rclpy.init(args=[])
    started = time.monotonic()
    try:
        collector = _SoakCollector(options.health_topic)
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
        startup_deadline = time.monotonic() + options.startup_timeout
        while time.monotonic() < startup_deadline and not collector.observed:
            rclpy.spin_once(collector, timeout_sec=0.1)
        if not collector.observed:
            raise RuntimeError("geometric hazard health did not become observable")
        started = time.monotonic()
        playback = subprocess.Popen(
            [
                "ros2",
                "bag",
                "play",
                str(options.bag),
                "--clock",
                "--loop",
                "--rate",
                str(options.rate),
                "--read-ahead-queue-size",
                "1000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = started + options.duration_seconds
        while time.monotonic() < deadline:
            if launch.poll() is not None:
                raise RuntimeError("geometric hazard launch stopped during soak")
            if playback.poll() is not None:
                stderr = playback.communicate()[1].decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(f"rosbag replay stopped early: {stderr.strip()}")
            rclpy.spin_once(collector, timeout_sec=0.2)
            now = time.monotonic()
            if now - collector.last_health_monotonic > 2.0:
                raise RuntimeError("geometric hazard health became stale during soak")
            if (
                collector.last_stage_progress_monotonic is None
                and now - started > 5.0
            ):
                raise RuntimeError(
                    "perception processing never started during soak"
                )
            if (
                collector.last_stage_progress_monotonic is not None
                and now - collector.last_stage_progress_monotonic > 5.0
            ):
                raise RuntimeError(
                    "perception processing stopped progressing during soak"
                )
    finally:
        elapsed = time.monotonic() - started
        if playback is not None:
            _stop(playback)
        if launch is not None:
            _stop(launch)
        if collector is not None:
            collector.destroy_node()
        rclpy.shutdown()
    assert collector is not None
    limits = BudgetLimits(
        processing_p95_ms=profile.resource_budget.processing_p95_ms,
        depth_geometry_average_cpu_cores=(
            profile.resource_budget.depth_geometry_average_cpu_cores
        ),
        soak_duration_seconds=profile.resource_budget.soak_duration_seconds,
        maximum_memory_growth_bytes=(
            profile.resource_budget.maximum_memory_growth_bytes
        ),
        maximum_pending_work=profile.resource_budget.maximum_input_queue_depth,
    )
    additional_reasons: list[str] = []
    if collector.profile_id != profile.profile_id:
        additional_reasons.append("profile_id_mismatch")
    if collector.profile_fingerprint != profile.fingerprint:
        additional_reasons.append("profile_fingerprint_mismatch")
    if collector.profile_mismatch_observed:
        additional_reasons.append("runtime_profile_mismatch")
    if (
        collector.maximum_rgb_reorder_pending
        > profile.resource_budget.maximum_rgb_reorder_depth
    ):
        additional_reasons.append("rgb_reorder_queue_unbounded")
    if (
        collector.rgb_reorder_capacity
        != profile.resource_budget.maximum_rgb_reorder_depth
    ):
        additional_reasons.append("rgb_reorder_capacity_mismatch")
    report = ResourceBudgetAudit(limits).evaluate(
        elapsed_seconds=elapsed,
        perception_processing_samples_ms=collector.processing_samples_ms,
        depth_geometry_average_cpu_cores=collector.depth_average_cpu_cores,
        resource_samples=collector.resource_samples,
        maximum_pending_work=collector.maximum_pending_work,
        frame_drops=collector.frame_drops,
        npu_state=collector.npu_state,
        additional_reasons=additional_reasons,
    )
    document = {
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "requested_duration_seconds": options.duration_seconds,
        "processing_sample_count": len(collector.processing_samples_ms),
        "resource_sample_count": len(collector.resource_samples),
        "maximum_rgb_reorder_pending": (
            collector.maximum_rgb_reorder_pending
        ),
        "rgb_reorder_capacity": collector.rgb_reorder_capacity,
        "report": asdict(report),
    }
    options.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)
