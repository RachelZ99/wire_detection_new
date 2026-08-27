"""Provider-independent seam into a robot-owned obstacle-response layer.

This module intentionally contains no chassis or navigation command.  A robot
integration supplies :class:`UnifiedObstacleResponsePort`; local tests use the
recording test double below.
"""

from __future__ import annotations

import math
import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

from .health import HealthState


class ResponseSourceMode(str, Enum):
    """The operational availability exposed to unified obstacle response."""

    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ObstacleResponseConfig:
    """Explicit provisional input policy; physical use requires approval."""

    expected_profile_id: str
    maximum_speed_mps: float
    health_timeout_ns: int
    maximum_observation_age_ns: int
    accept_confirmed_while_degraded: bool = True
    allow_clear_while_degraded: bool = False

    def __post_init__(self) -> None:
        if not self.expected_profile_id:
            raise ValueError("expected_profile_id is required")
        if not math.isfinite(self.maximum_speed_mps) or self.maximum_speed_mps <= 0:
            raise ValueError("maximum_speed_mps must be finite and positive")
        if self.health_timeout_ns <= 0 or self.maximum_observation_age_ns <= 0:
            raise ValueError("health and observation age limits must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "ObstacleResponseConfig":
        try:
            document = json.loads(Path(path).read_bytes())
            policy = document["provisional_input_policy"]
            profile = document["profile_binding"]
            return cls(
                expected_profile_id=str(profile["expected_profile_id"]),
                maximum_speed_mps=float(profile["maximum_speed_mps"]),
                health_timeout_ns=int(float(policy["health_timeout_ms"]) * 1_000_000),
                maximum_observation_age_ns=int(
                    float(policy["maximum_observation_age_ms"]) * 1_000_000
                ),
                accept_confirmed_while_degraded=bool(
                    policy["accept_confirmed_while_degraded"]
                ),
                allow_clear_while_degraded=bool(policy["allow_clear_while_degraded"]),
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"invalid obstacle-response config {path}: {error}"
            ) from error


@dataclass(frozen=True)
class PerceptionHealth:
    """Only the health/profile fields permitted at the downstream seam."""

    state: HealthState
    received_monotonic_ns: int
    heartbeat_stamp_ns: int
    profile_id: str
    profile_binding_state: str
    profile_maximum_speed_mps: float
    observed_speed_mps: float


def perception_health_from_values(
    *,
    state: str,
    values: Mapping[str, str],
    received_monotonic_ns: int,
    heartbeat_stamp_ns: int,
) -> PerceptionHealth:
    """Select the complete allow-list from a perception health diagnostic."""
    try:
        return PerceptionHealth(
            state=HealthState(state),
            received_monotonic_ns=received_monotonic_ns,
            heartbeat_stamp_ns=heartbeat_stamp_ns,
            profile_id=values["profile.id"],
            profile_binding_state=values["profile.binding_state"],
            profile_maximum_speed_mps=float(values["profile.maximum_speed_mps"]),
            observed_speed_mps=float(values["profile.latest_observed_speed_mps"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid operational perception health: {error}") from error


@dataclass(frozen=True)
class ConfirmedHazard:
    hazard_track_id: int
    observation_stamp_ns: int
    points_odom: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ConfirmedHazardSnapshot:
    """One complete operational snapshot from confirmed PointCloud2."""

    observation_stamp_ns: int
    hazards: tuple[ConfirmedHazard, ...]
    explicit_empty: bool


@dataclass(frozen=True)
class ResponseSourceStatus:
    mode: ResponseSourceMode
    top_level_health: HealthState | None
    reason: str
    source_generation: int
    awaiting_fresh_snapshot: bool


class UnifiedObstacleResponsePort(Protocol):
    """Adapter seam to be implemented against the robot's real interface."""

    def update_source_status(self, status: ResponseSourceStatus) -> None: ...

    def replace_confirmed_hazards(self, snapshot: ConfirmedHazardSnapshot) -> None: ...


class RecordingObstacleResponsePort:
    """In-memory test double; never sends a motion or navigation command."""

    def __init__(self) -> None:
        self.statuses: list[ResponseSourceStatus] = []
        self.snapshots: list[ConfirmedHazardSnapshot] = []

    def update_source_status(self, status: ResponseSourceStatus) -> None:
        if not self.statuses or self.statuses[-1] != status:
            self.statuses.append(status)

    def replace_confirmed_hazards(self, snapshot: ConfirmedHazardSnapshot) -> None:
        self.snapshots.append(snapshot)


class UnifiedObstacleResponseBridge:
    """Validate operational perception input before invoking the robot seam."""

    def __init__(
        self,
        config: ObstacleResponseConfig,
        port: UnifiedObstacleResponsePort,
    ) -> None:
        self.config = config
        self._port = port
        self._health: PerceptionHealth | None = None
        self._mode = ResponseSourceMode.BLOCKED
        self._reason = "health:missing"
        self._source_generation = 0
        self._awaiting_fresh_snapshot = True
        self._health_liveness_was_lost = False
        self._pending_snapshot: tuple[ConfirmedHazardSnapshot, int] | None = None

    def consume_health(self, health: PerceptionHealth) -> None:
        if health.received_monotonic_ns < 0 or health.heartbeat_stamp_ns < 0:
            raise ValueError("health timestamps cannot be negative")
        prior = self._health
        restarted = self._health_liveness_was_lost or (
            prior is not None and health.heartbeat_stamp_ns < prior.heartbeat_stamp_ns
        )
        if restarted:
            self._source_generation += 1
            self._awaiting_fresh_snapshot = True
        self._health_liveness_was_lost = False
        self._health = health
        self._apply_health_policy()
        self._publish_status()
        self._apply_pending_snapshot(sensor_now_ns=health.heartbeat_stamp_ns)

    def tick(self, *, received_monotonic_ns: int) -> None:
        if received_monotonic_ns < 0:
            raise ValueError("received_monotonic_ns cannot be negative")
        if self._health is None:
            self._mode = ResponseSourceMode.BLOCKED
            self._reason = "health:missing"
        elif (
            received_monotonic_ns - self._health.received_monotonic_ns
            > self.config.health_timeout_ns
        ):
            self._mode = ResponseSourceMode.BLOCKED
            self._reason = "health:stale"
            self._awaiting_fresh_snapshot = True
            self._health_liveness_was_lost = True
        else:
            self._apply_health_policy()
        self._publish_status()

    def consume_cloud(
        self,
        snapshot: ConfirmedHazardSnapshot,
        *,
        sensor_now_ns: int,
        received_monotonic_ns: int,
    ) -> bool:
        """Apply one snapshot; silence alone never removes retained hazards."""
        self.tick(received_monotonic_ns=received_monotonic_ns)
        if self._mode is ResponseSourceMode.BLOCKED:
            self._pending_snapshot = (snapshot, sensor_now_ns)
            return False
        return self._apply_snapshot(snapshot, sensor_now_ns=sensor_now_ns)

    def reject_operational_input(self, reason: str) -> None:
        """Block a malformed health/cloud contract without clearing hazards."""
        if not reason:
            raise ValueError("operational input rejection reason is required")
        self._awaiting_fresh_snapshot = True
        self._block(f"contract:{reason}")

    def _apply_snapshot(
        self,
        snapshot: ConfirmedHazardSnapshot,
        *,
        sensor_now_ns: int,
    ) -> bool:
        age_ns = sensor_now_ns - snapshot.observation_stamp_ns
        if age_ns < 0:
            self._block("cloud:observation_from_future")
            return False
        if age_ns > self.config.maximum_observation_age_ns:
            self._block("cloud:observation_stale")
            return False
        if snapshot.explicit_empty:
            if (
                self._mode is ResponseSourceMode.DEGRADED
                and not self.config.allow_clear_while_degraded
            ):
                return False
        else:
            if (
                self._mode is ResponseSourceMode.DEGRADED
                and not self.config.accept_confirmed_while_degraded
            ):
                return False
        self._port.replace_confirmed_hazards(snapshot)
        if self._awaiting_fresh_snapshot:
            self._awaiting_fresh_snapshot = False
            self._publish_status()
        return True

    def _apply_pending_snapshot(self, *, sensor_now_ns: int) -> None:
        pending = self._pending_snapshot
        if pending is None or self._mode is ResponseSourceMode.BLOCKED:
            return
        self._pending_snapshot = None
        snapshot, sensor_now_at_receive_ns = pending
        self._apply_snapshot(
            snapshot,
            sensor_now_ns=max(sensor_now_ns, sensor_now_at_receive_ns),
        )

    def _apply_health_policy(self) -> None:
        assert self._health is not None
        health = self._health
        profile_reason = self._profile_reason(health)
        if profile_reason:
            self._mode = ResponseSourceMode.BLOCKED
            self._reason = profile_reason
        elif health.state is HealthState.INVALID:
            self._mode = ResponseSourceMode.BLOCKED
            self._reason = "health:INVALID"
        elif health.state is HealthState.DEGRADED:
            self._mode = ResponseSourceMode.DEGRADED
            self._reason = "health:DEGRADED"
        else:
            self._mode = ResponseSourceMode.ACTIVE
            self._reason = "health:HEALTHY"

    def _profile_reason(self, health: PerceptionHealth) -> str:
        if health.profile_id != self.config.expected_profile_id:
            return "profile:id_mismatch"
        if health.profile_binding_state != "BOUND":
            return "profile:not_bound"
        values = (
            health.profile_maximum_speed_mps,
            health.observed_speed_mps,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            return "profile:speed_invalid"
        if (
            health.profile_maximum_speed_mps > self.config.maximum_speed_mps + 1e-9
            or health.observed_speed_mps > self.config.maximum_speed_mps + 1e-9
        ):
            return "profile:speed_exceeded"
        return ""

    def _block(self, reason: str) -> None:
        self._mode = ResponseSourceMode.BLOCKED
        self._reason = reason
        self._publish_status()

    def _publish_status(self) -> None:
        self._port.update_source_status(
            ResponseSourceStatus(
                mode=self._mode,
                top_level_health=(
                    self._health.state if self._health is not None else None
                ),
                reason=self._reason,
                source_generation=self._source_generation,
                awaiting_fresh_snapshot=self._awaiting_fresh_snapshot,
            )
        )


@dataclass(frozen=True)
class ConfirmedCloudView:
    """ROS-independent view of the PointCloud2 fields used downstream."""

    frame_id: str
    header_stamp_ns: int
    width: int
    height: int
    point_step: int
    row_step: int
    data: bytes
    fields: Mapping[str, tuple[int, str]]
    is_bigendian: bool = False


_REQUIRED_FIELDS = {
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "observation_stamp_sec": "int32",
    "observation_stamp_nanosec": "uint32",
    "cloud_group_index": "uint32",
    "hazard_track_id": "uint32",
}


def decode_confirmed_cloud(view: ConfirmedCloudView) -> ConfirmedHazardSnapshot:
    """Decode only the provider-independent operational PointCloud2 fields."""
    if view.frame_id != "odom":
        raise ValueError("confirmed cloud frame_id must be odom")
    if view.is_bigendian:
        raise ValueError("confirmed cloud must use the little-endian contract")
    if view.header_stamp_ns <= 0:
        raise ValueError("confirmed cloud header observation stamp must be positive")
    if view.width < 0 or view.height <= 0 or view.point_step <= 0:
        raise ValueError("confirmed cloud dimensions are invalid")
    for name, datatype in _REQUIRED_FIELDS.items():
        field_value = view.fields.get(name)
        if field_value is None:
            raise ValueError(f"confirmed cloud is missing {name}")
        if field_value[1] != datatype:
            raise ValueError(f"confirmed cloud field {name} must be {datatype}")
    point_count = view.width * view.height
    if point_count == 0:
        if view.data:
            raise ValueError("explicit empty cloud must not contain point data")
        return ConfirmedHazardSnapshot(
            observation_stamp_ns=view.header_stamp_ns,
            hazards=(),
            explicit_empty=True,
        )
    if view.row_step < view.width * view.point_step:
        raise ValueError("confirmed cloud row_step is shorter than one row")
    required_size = view.row_step * view.height
    if len(view.data) < required_size:
        raise ValueError("confirmed cloud data is shorter than declared dimensions")

    groups: dict[int, dict[str, object]] = {}
    for row in range(view.height):
        for column in range(view.width):
            base = row * view.row_step + column * view.point_step
            x = _unpack(view, "x", "<f", base)
            y = _unpack(view, "y", "<f", base)
            z = _unpack(view, "z", "<f", base)
            if not all(math.isfinite(value) for value in (x, y, z)):
                raise ValueError("confirmed cloud contains a non-finite odom point")
            seconds = int(_unpack(view, "observation_stamp_sec", "<i", base))
            nanoseconds = int(_unpack(view, "observation_stamp_nanosec", "<I", base))
            if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
                raise ValueError("confirmed cloud has an invalid observation stamp")
            stamp_ns = seconds * 1_000_000_000 + nanoseconds
            if stamp_ns <= 0:
                raise ValueError("confirmed cloud observation stamp must be positive")
            group_id = int(_unpack(view, "cloud_group_index", "<I", base))
            track_id = int(_unpack(view, "hazard_track_id", "<I", base))
            group = groups.setdefault(
                group_id,
                {"track_id": track_id, "stamp_ns": stamp_ns, "points": []},
            )
            if group["track_id"] != track_id or group["stamp_ns"] != stamp_ns:
                raise ValueError(
                    "one confirmed cloud group must have one track and observation stamp"
                )
            points = group["points"]
            assert isinstance(points, list)
            points.append((float(x), float(y), float(z)))
    hazards = tuple(
        ConfirmedHazard(
            hazard_track_id=int(group["track_id"]),
            observation_stamp_ns=int(group["stamp_ns"]),
            points_odom=tuple(group["points"]),
        )
        for _, group in sorted(groups.items())
    )
    oldest_stamp_ns = min(hazard.observation_stamp_ns for hazard in hazards)
    if view.header_stamp_ns != oldest_stamp_ns:
        raise ValueError(
            "confirmed cloud header must use the oldest point observation stamp"
        )
    return ConfirmedHazardSnapshot(
        observation_stamp_ns=oldest_stamp_ns,
        hazards=hazards,
        explicit_empty=False,
    )


def _unpack(
    view: ConfirmedCloudView, name: str, format_string: str, base: int
) -> int | float:
    offset = view.fields[name][0]
    if offset < 0 or offset + struct.calcsize(format_string) > view.point_step:
        raise ValueError(f"confirmed cloud field {name} is outside point_step")
    return struct.unpack_from(format_string, view.data, base + offset)[0]
