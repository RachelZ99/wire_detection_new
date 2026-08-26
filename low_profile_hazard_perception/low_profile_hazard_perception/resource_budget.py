"""Bounded runtime telemetry and evidence-backed resource-budget audits."""

from __future__ import annotations

import os
import resource
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageSnapshot:
    sample_count: int
    retained_sample_count: int
    processing_wall_latest_ms: float | None
    processing_wall_mean_ms: float | None
    processing_wall_p95_ms: float | None
    processing_cpu_mean_ms: float | None
    average_cpu_cores: float | None
    latest_queue_wait_ms: float | None
    latest_message_age_ms: float | None

    def diagnostic_values(self, prefix: str) -> dict[str, object]:
        return {f"{prefix}.{key}": value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class _StageSample:
    started_monotonic_ns: int
    completed_monotonic_ns: int
    cpu_time_ns: int
    queue_wait_ms: float
    message_age_ms: float

    @property
    def wall_ms(self) -> float:
        return (self.completed_monotonic_ns - self.started_monotonic_ns) / 1_000_000


class StageMetrics:
    """Retain a fixed-size measurement window while keeping a total count."""

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity < 2:
            raise ValueError("stage metric capacity must be at least two")
        self._samples: deque[_StageSample] = deque(maxlen=capacity)
        self._sample_count = 0

    def record(
        self,
        *,
        started_monotonic_ns: int,
        completed_monotonic_ns: int,
        cpu_time_ns: int,
        queue_wait_ms: float,
        message_age_ms: float,
    ) -> None:
        if completed_monotonic_ns < started_monotonic_ns:
            raise ValueError("stage completion precedes its start")
        if cpu_time_ns < 0 or queue_wait_ms < 0.0:
            raise ValueError("stage CPU time and queue wait cannot be negative")
        self._samples.append(
            _StageSample(
                started_monotonic_ns=started_monotonic_ns,
                completed_monotonic_ns=completed_monotonic_ns,
                cpu_time_ns=cpu_time_ns,
                queue_wait_ms=queue_wait_ms,
                message_age_ms=message_age_ms,
            )
        )
        self._sample_count += 1

    def snapshot(self) -> StageSnapshot:
        if not self._samples:
            return StageSnapshot(
                sample_count=self._sample_count,
                retained_sample_count=0,
                processing_wall_latest_ms=None,
                processing_wall_mean_ms=None,
                processing_wall_p95_ms=None,
                processing_cpu_mean_ms=None,
                average_cpu_cores=None,
                latest_queue_wait_ms=None,
                latest_message_age_ms=None,
            )
        samples = tuple(self._samples)
        wall = sorted(sample.wall_ms for sample in samples)
        cpu = [sample.cpu_time_ns / 1_000_000 for sample in samples]
        measurement_ns = (
            samples[-1].completed_monotonic_ns
            - samples[0].started_monotonic_ns
        )
        average_cores = (
            sum(sample.cpu_time_ns for sample in samples) / measurement_ns
            if measurement_ns > 0
            else None
        )
        return StageSnapshot(
            sample_count=self._sample_count,
            retained_sample_count=len(samples),
            processing_wall_latest_ms=samples[-1].wall_ms,
            processing_wall_mean_ms=sum(wall) / len(wall),
            processing_wall_p95_ms=_percentile(wall, 0.95),
            processing_cpu_mean_ms=sum(cpu) / len(cpu),
            average_cpu_cores=average_cores,
            latest_queue_wait_ms=samples[-1].queue_wait_ms,
            latest_message_age_ms=samples[-1].message_age_ms,
        )


@dataclass(frozen=True)
class ResourceSample:
    elapsed_seconds: float
    memory_rss_bytes: int
    cpu_cores: float


@dataclass(frozen=True)
class ProcessResourceSnapshot:
    elapsed_seconds: float
    cpu_cores: float | None
    memory_rss_bytes: int
    memory_peak_rss_bytes: int
    memory_growth_bytes: int
    retained_sample_count: int
    npu_state: str

    def diagnostic_values(self, prefix: str = "resource") -> dict[str, object]:
        return {f"{prefix}.{key}": value for key, value in asdict(self).items()}


class ProcessResourceMonitor:
    """Sample the current process without an optional psutil dependency."""

    def __init__(self, *, capacity: int = 720, npu_state: str) -> None:
        if capacity < 2:
            raise ValueError("resource sample capacity must be at least two")
        self._samples: deque[tuple[int, int, int]] = deque(maxlen=capacity)
        self._started_ns = time.monotonic_ns()
        self._initial_rss = _current_rss_bytes()
        self._npu_state = npu_state

    def sample(self) -> ProcessResourceSnapshot:
        monotonic_ns = time.monotonic_ns()
        process_ns = time.process_time_ns()
        rss = _current_rss_bytes()
        self._samples.append((monotonic_ns, process_ns, rss))
        cpu_cores = None
        if len(self._samples) >= 2:
            first = self._samples[0]
            last = self._samples[-1]
            elapsed_ns = last[0] - first[0]
            if elapsed_ns > 0:
                cpu_cores = max(0.0, (last[1] - first[1]) / elapsed_ns)
        return ProcessResourceSnapshot(
            elapsed_seconds=(monotonic_ns - self._started_ns) / 1_000_000_000,
            cpu_cores=cpu_cores,
            memory_rss_bytes=rss,
            memory_peak_rss_bytes=max(
                (sample[2] for sample in self._samples), default=rss
            ),
            memory_growth_bytes=rss - self._initial_rss,
            retained_sample_count=len(self._samples),
            npu_state=self._npu_state,
        )


@dataclass(frozen=True)
class BudgetLimits:
    processing_p95_ms: float
    depth_geometry_average_cpu_cores: float
    soak_duration_seconds: int
    maximum_memory_growth_bytes: int
    maximum_pending_work: int


@dataclass(frozen=True)
class ResourceBudgetReport:
    passed: bool
    reasons: tuple[str, ...]
    elapsed_seconds: float
    processing_p95_ms: float | None
    depth_geometry_average_cpu_cores: float | None
    memory_growth_bytes: int | None
    maximum_pending_work: int
    frame_drops: int
    npu_state: str


class ResourceBudgetAudit:
    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits

    def evaluate(
        self,
        *,
        elapsed_seconds: float,
        perception_processing_samples_ms: Iterable[float],
        depth_geometry_average_cpu_cores: float | None,
        resource_samples: Iterable[ResourceSample],
        maximum_pending_work: int,
        frame_drops: int,
        npu_state: str,
        additional_reasons: Iterable[str] = (),
    ) -> ResourceBudgetReport:
        processing = sorted(float(value) for value in perception_processing_samples_ms)
        resources = tuple(resource_samples)
        p95 = _percentile(processing, 0.95) if processing else None
        memory_growth = (
            resources[-1].memory_rss_bytes - resources[0].memory_rss_bytes
            if len(resources) >= 2
            else None
        )
        reasons: list[str] = []
        if p95 is None:
            reasons.append("processing_samples_missing")
        elif p95 > self.limits.processing_p95_ms:
            reasons.append("processing_p95_exceeded")
        if depth_geometry_average_cpu_cores is None:
            reasons.append("depth_cpu_samples_missing")
        elif (
            depth_geometry_average_cpu_cores
            > self.limits.depth_geometry_average_cpu_cores
        ):
            reasons.append("depth_cpu_budget_exceeded")
        if elapsed_seconds < self.limits.soak_duration_seconds:
            reasons.append("soak_duration_incomplete")
        if memory_growth is None:
            reasons.append("memory_samples_missing")
        elif memory_growth > self.limits.maximum_memory_growth_bytes:
            reasons.append("memory_growth_exceeded")
        if maximum_pending_work > self.limits.maximum_pending_work:
            reasons.append("pending_work_unbounded")
        reasons.extend(str(reason) for reason in additional_reasons)
        return ResourceBudgetReport(
            passed=not reasons,
            reasons=tuple(reasons),
            elapsed_seconds=elapsed_seconds,
            processing_p95_ms=p95,
            depth_geometry_average_cpu_cores=depth_geometry_average_cpu_cores,
            memory_growth_bytes=memory_growth,
            maximum_pending_work=maximum_pending_work,
            frame_drops=frame_drops,
            npu_state=npu_state,
        )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def _current_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
