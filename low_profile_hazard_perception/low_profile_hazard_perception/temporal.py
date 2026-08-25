"""Observation-time odometry alignment and two-observation confirmation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
from math import acos, degrees, sin, sqrt

from .geometry import Point3


Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class OdomAlignment:
    pose: Pose3 | None
    reason: str = ""


@dataclass(frozen=True)
class Pose3:
    """Rigid transform whose translation and quaternion map local to parent."""

    translation: Point3
    rotation: Quaternion

    @staticmethod
    def identity() -> "Pose3":
        return Pose3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def normalized(self) -> "Pose3":
        return Pose3(self.translation, _normalize(self.rotation))

    def transform_point(self, point: Point3) -> Point3:
        rotated = _rotate(self.rotation, point)
        return tuple(
            rotated[index] + self.translation[index] for index in range(3)
        )

    def compose(self, local: "Pose3") -> "Pose3":
        """Return this transform followed by a child-to-local transform."""
        return Pose3(
            translation=self.transform_point(local.translation),
            rotation=_multiply(self.rotation, local.rotation),
        ).normalized()

    def inverse(self) -> "Pose3":
        normalized = _normalize(self.rotation)
        inverse_rotation = (
            -normalized[0],
            -normalized[1],
            -normalized[2],
            normalized[3],
        )
        inverse_translation = _rotate(
            inverse_rotation,
            tuple(-value for value in self.translation),
        )
        return Pose3(inverse_translation, inverse_rotation)

    def with_observed_ground(
        self, normal_local: Point3, camera_height_m: float
    ) -> "Pose3":
        """Correct nominal tilt/height with the currently observed floor."""
        normal_parent = _rotate(self.rotation, normal_local)
        correction = _rotation_between(normal_parent, (0.0, 0.0, 1.0))
        return Pose3(
            translation=(
                self.translation[0],
                self.translation[1],
                camera_height_m,
            ),
            rotation=_multiply(correction, self.rotation),
        ).normalized()

    def normal_error_to_parent_up_degrees(self, normal_local: Point3) -> float:
        normal_parent = _normalize_vector(_rotate(self.rotation, normal_local))
        return degrees(acos(max(-1.0, min(1.0, normal_parent[2]))))


class OdomPoseCache:
    """Interpolate translation/quaternion in a bounded timestamp cache."""

    def __init__(
        self,
        *,
        maximum_samples: int = 512,
        maximum_age_ns: int = 5_000_000_000,
        maximum_interpolation_gap_ns: int = 100_000_000,
        maximum_translation_jump_m: float = 0.25,
        maximum_rotation_jump_degrees: float = 45.0,
    ) -> None:
        if maximum_samples < 2:
            raise ValueError("maximum_samples must be at least two")
        if maximum_age_ns <= 0 or maximum_interpolation_gap_ns <= 0:
            raise ValueError("odom cache time bounds must be positive")
        if maximum_translation_jump_m <= 0.0:
            raise ValueError("maximum_translation_jump_m must be positive")
        if not 0.0 < maximum_rotation_jump_degrees <= 180.0:
            raise ValueError(
                "maximum_rotation_jump_degrees must be in (0, 180]"
            )
        self._maximum_samples = maximum_samples
        self._maximum_age_ns = maximum_age_ns
        self._maximum_interpolation_gap_ns = maximum_interpolation_gap_ns
        self._maximum_translation_jump_m = maximum_translation_jump_m
        self._maximum_rotation_jump_degrees = maximum_rotation_jump_degrees
        self._stamps: list[int] = []
        self._poses: list[Pose3] = []
        self._last_arrival_stamp_ns: int | None = None

    @property
    def latest_stamp_ns(self) -> int | None:
        return self._stamps[-1] if self._stamps else None

    @property
    def maximum_interpolation_gap_ns(self) -> int:
        return self._maximum_interpolation_gap_ns

    @property
    def maximum_translation_jump_m(self) -> float:
        return self._maximum_translation_jump_m

    @property
    def maximum_rotation_jump_degrees(self) -> float:
        return self._maximum_rotation_jump_degrees

    def add(self, sensor_stamp_ns: int, pose: Pose3) -> str:
        if sensor_stamp_ns <= 0:
            raise ValueError("odom sensor stamp must be positive")
        if (
            self._last_arrival_stamp_ns is not None
            and sensor_stamp_ns < self._last_arrival_stamp_ns
        ):
            return "odom:disordered"
        pose = pose.normalized()
        self._last_arrival_stamp_ns = sensor_stamp_ns
        index = bisect_left(self._stamps, sensor_stamp_ns)
        if (
            index < len(self._stamps)
            and self._stamps[index] == sensor_stamp_ns
        ):
            self._poses[index] = pose
        else:
            self._stamps.insert(index, sensor_stamp_ns)
            self._poses.insert(index, pose)
        newest = self._stamps[-1]
        first_allowed = newest - self._maximum_age_ns
        trim = bisect_left(self._stamps, first_allowed)
        if trim:
            del self._stamps[:trim]
            del self._poses[:trim]
        overflow = len(self._stamps) - self._maximum_samples
        if overflow > 0:
            del self._stamps[:overflow]
            del self._poses[:overflow]
        return ""

    def interpolate(self, sensor_stamp_ns: int) -> Pose3 | None:
        return self.alignment_at(sensor_stamp_ns).pose

    def alignment_at(self, sensor_stamp_ns: int) -> OdomAlignment:
        if len(self._stamps) < 2:
            return OdomAlignment(None, "odom:missing")
        upper = bisect_left(self._stamps, sensor_stamp_ns)
        if (
            upper < len(self._stamps)
            and self._stamps[upper] == sensor_stamp_ns
        ):
            supported = (
                upper > 0 and self._segment_is_supported(upper - 1, upper)
            ) or (
                upper + 1 < len(self._stamps)
                and self._segment_is_supported(upper, upper + 1)
            )
            if supported:
                return OdomAlignment(self._poses[upper])
            return OdomAlignment(None, "odom:discontinuous")
        if upper == 0 or upper == len(self._stamps):
            return OdomAlignment(None, "odom:stale")
        lower = upper - 1
        if not self._segment_is_supported(lower, upper):
            return OdomAlignment(None, "odom:discontinuous")
        interval = self._stamps[upper] - self._stamps[lower]
        fraction = (sensor_stamp_ns - self._stamps[lower]) / interval
        first = self._poses[lower]
        second = self._poses[upper]
        translation = tuple(
            first.translation[index]
            + fraction * (second.translation[index] - first.translation[index])
            for index in range(3)
        )
        return OdomAlignment(
            Pose3(
                translation=translation,
                rotation=_slerp(first.rotation, second.rotation, fraction),
            )
        )

    def continuous_between(
        self, first_stamp_ns: int, second_stamp_ns: int
    ) -> bool:
        """Check every odom segment spanning two observation stamps."""
        if first_stamp_ns > second_stamp_ns:
            first_stamp_ns, second_stamp_ns = (
                second_stamp_ns,
                first_stamp_ns,
            )
        first_upper = bisect_left(self._stamps, first_stamp_ns)
        first_index = first_upper
        if (
            first_upper == len(self._stamps)
            or self._stamps[first_upper] != first_stamp_ns
        ):
            first_index -= 1
        second_index = bisect_left(self._stamps, second_stamp_ns)
        if first_index < 0 or second_index >= len(self._stamps):
            return False
        return all(
            self._segment_is_supported(index, index + 1)
            for index in range(first_index, second_index)
        )

    def _segment_is_supported(self, lower: int, upper: int) -> bool:
        interval = self._stamps[upper] - self._stamps[lower]
        if interval <= 0 or interval > self._maximum_interpolation_gap_ns:
            return False
        first = self._poses[lower]
        second = self._poses[upper]
        translation_jump = sqrt(
            sum(
                (second.translation[index] - first.translation[index]) ** 2
                for index in range(3)
            )
        )
        rotation_dot = abs(
            sum(
                left * right
                for left, right in zip(
                    first.rotation, second.rotation, strict=True
                )
            )
        )
        rotation_jump = degrees(2.0 * acos(max(-1.0, min(1.0, rotation_dot))))
        return (
            translation_jump <= self._maximum_translation_jump_m
            and rotation_jump <= self._maximum_rotation_jump_degrees
        )


class EvidenceSource(str, Enum):
    STRONG_GEOMETRY = "STRONG_GEOMETRY"


@dataclass(frozen=True)
class HazardObservation:
    sensor_stamp_ns: int
    points_odom: tuple[Point3, ...]
    evidence: EvidenceSource
    confidence: float

    @property
    def centroid(self) -> Point3:
        if not self.points_odom:
            raise ValueError("hazard observation needs at least one point")
        return tuple(
            sum(point[axis] for point in self.points_odom)
            / len(self.points_odom)
            for axis in range(3)
        )


@dataclass(frozen=True)
class ConfirmedHazard:
    sensor_stamp_ns: int
    points_odom: tuple[Point3, ...]
    centroid: Point3
    observation_count: int
    evidence: tuple[EvidenceSource, ...]
    confidence: float
    spatial_spread_m: float


@dataclass(frozen=True)
class HazardTrackerConfig:
    association_radius_m: float = 0.08
    confirmation_window_ns: int = 350_000_000
    candidate_retention_ns: int = 500_000_000
    confirmed_retention_ns: int = 2_000_000_000

    def __post_init__(self) -> None:
        if self.association_radius_m <= 0.0:
            raise ValueError("association_radius_m must be positive")
        if self.confirmation_window_ns <= 0:
            raise ValueError("confirmation_window_ns must be positive")
        if self.candidate_retention_ns <= 0:
            raise ValueError("candidate_retention_ns must be positive")
        if self.confirmed_retention_ns < 2_000_000_000:
            raise ValueError(
                "confirmed_retention_ns must be at least two seconds"
            )


@dataclass
class _Track:
    first_stamp_ns: int
    last_stamp_ns: int
    observation_centroids: list[Point3]
    latest_points: tuple[Point3, ...]
    evidence: set[EvidenceSource] = field(default_factory=set)
    confidence: float = 0.0
    confirmed: bool = False

    @property
    def centroid(self) -> Point3:
        return tuple(
            sum(point[axis] for point in self.observation_centroids)
            / len(self.observation_centroids)
            for axis in range(3)
        )


class HazardTracker:
    """Associate in odom and expose only twice-observed hazards."""

    def __init__(self, config: HazardTrackerConfig | None = None) -> None:
        self.config = config or HazardTrackerConfig()
        self._tracks: list[_Track] = []

    def clear(self) -> None:
        self._tracks.clear()

    def clear_candidates(self) -> None:
        """Discard unconfirmed accumulation without clearing known hazards."""
        self._tracks = [track for track in self._tracks if track.confirmed]

    def retained_at(
        self,
        sensor_now_ns: int,
        *,
        allow_confirmed_expiry: bool = True,
    ) -> tuple[ConfirmedHazard, ...]:
        """Return confirmed hazards still retained at a sensor-clock time."""
        self._expire_at(
            sensor_now_ns,
            expire_confirmed=allow_confirmed_expiry,
        )
        return tuple(
            self._confirmed_hazard(track)
            for track in self._tracks
            if track.confirmed
        )

    def candidate_count_at(self, sensor_now_ns: int) -> int:
        """Return live unconfirmed tracks using candidate expiry only."""
        self._expire_at(sensor_now_ns, expire_confirmed=False)
        return sum(not track.confirmed for track in self._tracks)

    def observe(
        self,
        observation: HazardObservation,
        *,
        allow_confirmed_expiry: bool = True,
    ) -> tuple[ConfirmedHazard, ...]:
        if observation.sensor_stamp_ns <= 0:
            return ()
        centroid = observation.centroid
        self._expire_at(
            observation.sensor_stamp_ns,
            expire_confirmed=allow_confirmed_expiry,
        )
        matching = [
            (self._distance(centroid, track.centroid), track)
            for track in self._tracks
            if track.last_stamp_ns < observation.sensor_stamp_ns
            and (
                track.confirmed
                or observation.sensor_stamp_ns - track.last_stamp_ns
                <= self.config.confirmation_window_ns
            )
        ]
        matching = [
            item
            for item in matching
            if item[0] <= self.config.association_radius_m
        ]
        if matching:
            _, track = min(matching, key=lambda item: item[0])
            track.last_stamp_ns = observation.sensor_stamp_ns
            track.observation_centroids.append(centroid)
            track.latest_points = observation.points_odom
            track.evidence.add(observation.evidence)
            track.confidence = max(track.confidence, observation.confidence)
        else:
            track = _Track(
                first_stamp_ns=observation.sensor_stamp_ns,
                last_stamp_ns=observation.sensor_stamp_ns,
                observation_centroids=[centroid],
                latest_points=observation.points_odom,
                evidence={observation.evidence},
                confidence=observation.confidence,
            )
            self._tracks.append(track)
        if len(track.observation_centroids) < 2:
            return ()
        track.confirmed = True
        return (self._confirmed_hazard(track),)

    def _expire_at(
        self, sensor_now_ns: int, *, expire_confirmed: bool = True
    ) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if (
                track.confirmed
                and not expire_confirmed
                or sensor_now_ns - track.last_stamp_ns
                <= (
                    self.config.confirmed_retention_ns
                    if track.confirmed
                    else self.config.candidate_retention_ns
                )
            )
        ]

    def _confirmed_hazard(self, track: _Track) -> ConfirmedHazard:
        spread = max(
            self._distance(point, track.centroid)
            for point in track.observation_centroids
        )
        return ConfirmedHazard(
            sensor_stamp_ns=track.last_stamp_ns,
            points_odom=track.latest_points,
            centroid=track.centroid,
            observation_count=len(track.observation_centroids),
            evidence=tuple(
                sorted(track.evidence, key=lambda item: item.value)
            ),
            confidence=track.confidence,
            spatial_spread_m=spread,
        )

    @staticmethod
    def _distance(first: Point3, second: Point3) -> float:
        return sqrt(
            sum((first[index] - second[index]) ** 2 for index in range(3))
        )


def _normalize(quaternion: Quaternion) -> Quaternion:
    norm = sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion has zero norm")
    return tuple(value / norm for value in quaternion)


def _normalize_vector(vector: Point3) -> Point3:
    norm = sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("vector has zero norm")
    return tuple(value / norm for value in vector)


def _rotation_between(first: Point3, second: Point3) -> Quaternion:
    first = _normalize_vector(first)
    second = _normalize_vector(second)
    dot = max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(first, second, strict=True))),
    )
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    if dot < -0.999999:
        reference = (1.0, 0.0, 0.0)
        if abs(first[0]) > 0.9:
            reference = (0.0, 1.0, 0.0)
        axis = _normalize_vector(
            (
                first[1] * reference[2] - first[2] * reference[1],
                first[2] * reference[0] - first[0] * reference[2],
                first[0] * reference[1] - first[1] * reference[0],
            )
        )
        return axis[0], axis[1], axis[2], 0.0
    return _normalize((cross[0], cross[1], cross[2], 1.0 + dot))


def _multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate(quaternion: Quaternion, point: Point3) -> Point3:
    q = _normalize(quaternion)
    vector = (point[0], point[1], point[2], 0.0)
    conjugate = (-q[0], -q[1], -q[2], q[3])
    rotated = _multiply(_multiply(q, vector), conjugate)
    return rotated[0], rotated[1], rotated[2]


def _slerp(
    first: Quaternion, second: Quaternion, fraction: float
) -> Quaternion:
    first = _normalize(first)
    second = _normalize(second)
    dot = sum(left * right for left, right in zip(first, second, strict=True))
    if dot < 0.0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize(
            tuple(
                first[index] + fraction * (second[index] - first[index])
                for index in range(4)
            )
        )
    angle = acos(dot)
    scale = sin(angle)
    first_weight = sin((1.0 - fraction) * angle) / scale
    second_weight = sin(fraction * angle) / scale
    return tuple(
        first_weight * first[index] + second_weight * second[index]
        for index in range(4)
    )
