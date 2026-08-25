"""Observation-time odometry alignment and two-observation confirmation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from math import acos, sin, sqrt

from .geometry import Point3


Quaternion = tuple[float, float, float, float]


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


class OdomPoseCache:
    """Interpolate translation/quaternion in a bounded timestamp cache."""

    def __init__(
        self,
        *,
        maximum_samples: int = 512,
        maximum_age_ns: int = 5_000_000_000,
    ) -> None:
        if maximum_samples < 2:
            raise ValueError("maximum_samples must be at least two")
        self._maximum_samples = maximum_samples
        self._maximum_age_ns = maximum_age_ns
        self._stamps: list[int] = []
        self._poses: list[Pose3] = []

    @property
    def latest_stamp_ns(self) -> int | None:
        return self._stamps[-1] if self._stamps else None

    def add(self, sensor_stamp_ns: int, pose: Pose3) -> None:
        if sensor_stamp_ns <= 0:
            raise ValueError("odom sensor stamp must be positive")
        pose = pose.normalized()
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

    def interpolate(self, sensor_stamp_ns: int) -> Pose3 | None:
        if len(self._stamps) < 2:
            return None
        upper = bisect_left(self._stamps, sensor_stamp_ns)
        if (
            upper < len(self._stamps)
            and self._stamps[upper] == sensor_stamp_ns
        ):
            return self._poses[upper]
        if upper == 0 or upper == len(self._stamps):
            return None
        lower = upper - 1
        interval = self._stamps[upper] - self._stamps[lower]
        if interval <= 0:
            return None
        fraction = (sensor_stamp_ns - self._stamps[lower]) / interval
        first = self._poses[lower]
        second = self._poses[upper]
        translation = tuple(
            first.translation[index]
            + fraction * (second.translation[index] - first.translation[index])
            for index in range(3)
        )
        return Pose3(
            translation=translation,
            rotation=_slerp(first.rotation, second.rotation, fraction),
        )


@dataclass(frozen=True)
class HazardObservation:
    sensor_stamp_ns: int
    points_odom: tuple[Point3, ...]
    evidence: str
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
    evidence: tuple[str, ...]
    confidence: float
    spatial_spread_m: float


@dataclass(frozen=True)
class HazardTrackerConfig:
    association_radius_m: float = 0.08
    confirmation_window_ns: int = 350_000_000
    required_observations: int = 2

    def __post_init__(self) -> None:
        if self.association_radius_m <= 0.0:
            raise ValueError("association_radius_m must be positive")
        if self.required_observations < 2:
            raise ValueError("required_observations must be at least two")


@dataclass
class _Track:
    first_stamp_ns: int
    last_stamp_ns: int
    observation_centroids: list[Point3]
    latest_points: tuple[Point3, ...]
    evidence: set[str] = field(default_factory=set)
    confidence: float = 0.0

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

    def observe(
        self, observation: HazardObservation
    ) -> tuple[ConfirmedHazard, ...]:
        if observation.sensor_stamp_ns <= 0:
            return ()
        centroid = observation.centroid
        self._tracks = [
            track
            for track in self._tracks
            if observation.sensor_stamp_ns - track.last_stamp_ns
            <= self.config.confirmation_window_ns
        ]
        matching = [
            (self._distance(centroid, track.centroid), track)
            for track in self._tracks
            if track.last_stamp_ns < observation.sensor_stamp_ns
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
        if (
            len(track.observation_centroids)
            < self.config.required_observations
        ):
            return ()
        spread = max(
            self._distance(point, track.centroid)
            for point in track.observation_centroids
        )
        return (
            ConfirmedHazard(
                sensor_stamp_ns=observation.sensor_stamp_ns,
                points_odom=track.latest_points,
                centroid=track.centroid,
                observation_count=len(track.observation_centroids),
                evidence=tuple(sorted(track.evidence)),
                confidence=track.confidence,
                spatial_spread_m=spread,
            ),
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
