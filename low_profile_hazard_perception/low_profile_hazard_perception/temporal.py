"""Observation-time odometry alignment and two-observation confirmation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum, IntFlag
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
        return tuple(rotated[index] + self.translation[index] for index in range(3))

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
            raise ValueError("maximum_rotation_jump_degrees must be in (0, 180]")
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
        if index < len(self._stamps) and self._stamps[index] == sensor_stamp_ns:
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
        if upper < len(self._stamps) and self._stamps[upper] == sensor_stamp_ns:
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

    def continuous_between(self, first_stamp_ns: int, second_stamp_ns: int) -> bool:
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
                for left, right in zip(first.rotation, second.rotation, strict=True)
            )
        )
        rotation_jump = degrees(2.0 * acos(max(-1.0, min(1.0, rotation_dot))))
        return (
            translation_jump <= self._maximum_translation_jump_m
            and rotation_jump <= self._maximum_rotation_jump_degrees
        )


class EvidenceSource(str, Enum):
    STRONG_GEOMETRY = "STRONG_GEOMETRY"
    RGB_CABLE = "RGB_CABLE"
    WEAK_HEIGHT = "WEAK_HEIGHT"
    INVALID_DEPTH = "INVALID_DEPTH"


class EvidenceMask(IntFlag):
    """Stable PointCloud2 wire bits for operational evidence sources."""

    NONE = 0
    STRONG_GEOMETRY = 1
    RGB_CABLE = 2
    WEAK_HEIGHT = 4
    INVALID_DEPTH = 8

    @classmethod
    def from_sources(cls, sources: tuple[EvidenceSource, ...]) -> "EvidenceMask":
        mask = cls.NONE
        if EvidenceSource.STRONG_GEOMETRY in sources:
            mask |= cls.STRONG_GEOMETRY
        if EvidenceSource.RGB_CABLE in sources:
            mask |= cls.RGB_CABLE
        if EvidenceSource.WEAK_HEIGHT in sources:
            mask |= cls.WEAK_HEIGHT
        if EvidenceSource.INVALID_DEPTH in sources:
            mask |= cls.INVALID_DEPTH
        return mask


class CandidateDecisionReason(str, Enum):
    INVALID_SENSOR_STAMP = "INVALID_SENSOR_STAMP"
    DUPLICATE_OBSERVATION = "DUPLICATE_OBSERVATION"
    SUPPORT_ONLY = "SUPPORT_ONLY"
    LOW_CONFIDENCE_RGB = "LOW_CONFIDENCE_RGB"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    REJECTED_INSUFFICIENT_SUPPORT = "REJECTED_INSUFFICIENT_SUPPORT"
    REJECTED_INSUFFICIENT_SPAN = "REJECTED_INSUFFICIENT_SPAN"
    REJECTED_INVALID_DEPTH_TOO_WIDE = "REJECTED_INVALID_DEPTH_TOO_WIDE"
    REJECTED_INVALID_DEPTH_NOT_ENCLOSED = "REJECTED_INVALID_DEPTH_NOT_ENCLOSED"
    CONFIRMED_STRONG_GEOMETRY = "CONFIRMED_STRONG_GEOMETRY"
    CONFIRMED_RGB_CABLE = "CONFIRMED_RGB_CABLE"
    CONFIRMED_MIXED_EVIDENCE = "CONFIRMED_MIXED_EVIDENCE"


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
            sum(point[axis] for point in self.points_odom) / len(self.points_odom)
            for axis in range(3)
        )


@dataclass(frozen=True)
class ConfirmedHazard:
    hazard_track_id: int
    sensor_stamp_ns: int
    points_odom: tuple[Point3, ...]
    centroid: Point3
    observation_count: int
    evidence: tuple[EvidenceSource, ...]
    confidence: float
    spatial_spread_m: float
    confirmation_latency_ns: int


@dataclass(frozen=True)
class TrackingDecision:
    confirmed: tuple[ConfirmedHazard, ...]
    evidence: tuple[EvidenceSource, ...]
    confidence: float
    decision_reason: CandidateDecisionReason


@dataclass(frozen=True)
class HazardTrackerConfig:
    association_radius_m: float = 0.08
    confirmation_window_ns: int = 350_000_000
    candidate_retention_ns: int = 500_000_000
    confirmed_retention_ns: int = 2_000_000_000
    minimum_rgb_confirmation_confidence: float = 0.75

    def __post_init__(self) -> None:
        if self.association_radius_m <= 0.0:
            raise ValueError("association_radius_m must be positive")
        if self.confirmation_window_ns <= 0:
            raise ValueError("confirmation_window_ns must be positive")
        if self.candidate_retention_ns <= 0:
            raise ValueError("candidate_retention_ns must be positive")
        if self.confirmed_retention_ns < 2_000_000_000:
            raise ValueError("confirmed_retention_ns must be at least two seconds")
        if not 0.0 <= self.minimum_rgb_confirmation_confidence <= 1.0:
            raise ValueError("minimum_rgb_confirmation_confidence must be in [0, 1]")


@dataclass
class _Track:
    hazard_track_id: int
    first_stamp_ns: int
    last_stamp_ns: int
    observation_centroids: list[Point3]
    latest_points: tuple[Point3, ...]
    latest_points_stamp_ns: int
    support_centroids: list[Point3] = field(default_factory=list)
    confirmation_stamps: set[int] = field(default_factory=set)
    observation_keys: set[tuple[int, EvidenceSource, Point3]] = field(
        default_factory=set
    )
    evidence: set[EvidenceSource] = field(default_factory=set)
    confidence: float = 0.0
    confirmed: bool = False
    confirmation_latency_ns: int = 0
    refresh_centroids: list[Point3] = field(default_factory=list)
    refresh_last_stamp_ns: int | None = None
    refresh_latest_points: tuple[Point3, ...] = ()
    refresh_latest_points_stamp_ns: int | None = None
    refresh_stamps: set[int] = field(default_factory=set)
    refresh_evidence: set[EvidenceSource] = field(default_factory=set)
    refresh_confidence: float = 0.0

    @property
    def centroid(self) -> Point3:
        points = sorted(self.observation_centroids or self.support_centroids)
        return tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )


class HazardTracker:
    """Associate in odom and expose only twice-observed hazards."""

    def __init__(self, config: HazardTrackerConfig | None = None) -> None:
        self.config = config or HazardTrackerConfig()
        self._tracks: list[_Track] = []
        self._latest_sensor_stamp_ns = 0
        self._next_hazard_track_id = 0

    def clear(self) -> None:
        self._tracks.clear()
        self._latest_sensor_stamp_ns = 0
        self._next_hazard_track_id = 0

    def clear_candidates(self) -> None:
        """Discard unconfirmed accumulation without clearing known hazards."""
        self._tracks = [track for track in self._tracks if track.confirmed]
        for track in self._tracks:
            self._clear_refresh(track)

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
            self._confirmed_hazard(track) for track in self._tracks if track.confirmed
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
        require_reconfirmation_for_confirmed: bool = False,
    ) -> tuple[ConfirmedHazard, ...]:
        return self.observe_with_decision(
            observation,
            allow_confirmed_expiry=allow_confirmed_expiry,
            require_reconfirmation_for_confirmed=(require_reconfirmation_for_confirmed),
        ).confirmed

    def observe_with_decision(
        self,
        observation: HazardObservation,
        *,
        allow_confirmed_expiry: bool = True,
        require_reconfirmation_for_confirmed: bool = False,
    ) -> TrackingDecision:
        if observation.sensor_stamp_ns <= 0:
            return TrackingDecision(
                confirmed=(),
                evidence=(observation.evidence,),
                confidence=observation.confidence,
                decision_reason=CandidateDecisionReason.INVALID_SENSOR_STAMP,
            )
        centroid = observation.centroid
        can_confirm = self._can_confirm(observation)
        self._latest_sensor_stamp_ns = max(
            self._latest_sensor_stamp_ns, observation.sensor_stamp_ns
        )
        self._expire_at(
            self._latest_sensor_stamp_ns,
            expire_confirmed=allow_confirmed_expiry,
        )
        if require_reconfirmation_for_confirmed:
            confirmed_matching = [
                (self._distance(centroid, track.centroid), track)
                for track in self._tracks
                if track.confirmed
                and self._can_fuse_with_track(track)
                and self._distance(centroid, track.centroid)
                <= self.config.association_radius_m
            ]
            if confirmed_matching:
                _, track = min(confirmed_matching, key=lambda item: item[0])
                confirmed = self._observe_refresh(track, observation)
                return self._decision(
                    track,
                    confirmed,
                    (
                        self._confirmation_reason(track)
                        if confirmed
                        else self._pending_reason(observation, can_confirm)
                    ),
                )
        matching = [
            (self._distance(centroid, track.centroid), track)
            for track in self._tracks
            if self._can_fuse_with_track(track)
            and (not track.confirmed or not require_reconfirmation_for_confirmed)
            and (
                track.confirmed
                or max(track.last_stamp_ns, observation.sensor_stamp_ns)
                - min(track.first_stamp_ns, observation.sensor_stamp_ns)
                <= self.config.confirmation_window_ns
            )
        ]
        matching = [
            item for item in matching if item[0] <= self.config.association_radius_m
        ]
        if matching:
            _, track = min(matching, key=lambda item: item[0])
            observation_key = (
                observation.sensor_stamp_ns,
                observation.evidence,
                centroid,
            )
            if observation_key in track.observation_keys:
                return self._decision(
                    track,
                    (),
                    CandidateDecisionReason.DUPLICATE_OBSERVATION,
                )
            track.observation_keys.add(observation_key)
            track.first_stamp_ns = min(
                track.first_stamp_ns, observation.sensor_stamp_ns
            )
            track.last_stamp_ns = max(track.last_stamp_ns, observation.sensor_stamp_ns)
            counts_for_confirmation = (
                can_confirm
                and observation.sensor_stamp_ns not in track.confirmation_stamps
            )
            if not can_confirm:
                track.support_centroids.append(centroid)
            elif counts_for_confirmation:
                track.observation_centroids.append(centroid)
                track.confirmation_stamps.add(observation.sensor_stamp_ns)
                if observation.sensor_stamp_ns >= track.latest_points_stamp_ns:
                    track.latest_points = observation.points_odom
                    track.latest_points_stamp_ns = observation.sensor_stamp_ns
            track.evidence.add(observation.evidence)
            track.confidence = max(track.confidence, observation.confidence)
        else:
            counts_for_confirmation = can_confirm
            track = _Track(
                hazard_track_id=self._next_hazard_track_id,
                first_stamp_ns=observation.sensor_stamp_ns,
                last_stamp_ns=observation.sensor_stamp_ns,
                observation_centroids=[] if not can_confirm else [centroid],
                latest_points=observation.points_odom,
                latest_points_stamp_ns=observation.sensor_stamp_ns,
                support_centroids=[centroid] if not can_confirm else [],
                confirmation_stamps=(
                    {observation.sensor_stamp_ns} if can_confirm else set()
                ),
                observation_keys={
                    (
                        observation.sensor_stamp_ns,
                        observation.evidence,
                        centroid,
                    )
                },
                evidence={observation.evidence},
                confidence=observation.confidence,
            )
            self._next_hazard_track_id += 1
            self._tracks.append(track)
        if len(track.observation_centroids) < 2 or not counts_for_confirmation:
            reason = self._pending_reason(observation, can_confirm)
            return self._decision(track, (), reason)
        track.confirmed = True
        track.confirmation_latency_ns = (
            max(track.confirmation_stamps) - min(track.confirmation_stamps)
        )
        confirmed = (self._confirmed_hazard(track),)
        return self._decision(track, confirmed, self._confirmation_reason(track))

    def _decision(
        self,
        track: _Track,
        confirmed: tuple[ConfirmedHazard, ...],
        reason: CandidateDecisionReason,
    ) -> TrackingDecision:
        return TrackingDecision(
            confirmed=confirmed,
            evidence=tuple(sorted(track.evidence, key=lambda item: item.value)),
            confidence=self._track_confidence(track),
            decision_reason=reason,
        )

    @staticmethod
    def _confirmation_reason(track: _Track) -> CandidateDecisionReason:
        if track.evidence == {EvidenceSource.STRONG_GEOMETRY}:
            return CandidateDecisionReason.CONFIRMED_STRONG_GEOMETRY
        if track.evidence == {EvidenceSource.RGB_CABLE}:
            return CandidateDecisionReason.CONFIRMED_RGB_CABLE
        return CandidateDecisionReason.CONFIRMED_MIXED_EVIDENCE

    @staticmethod
    def _is_support(evidence: EvidenceSource) -> bool:
        return evidence in (
            EvidenceSource.WEAK_HEIGHT,
            EvidenceSource.INVALID_DEPTH,
        )

    def _can_confirm(self, observation: HazardObservation) -> bool:
        if observation.evidence is EvidenceSource.STRONG_GEOMETRY:
            return True
        return (
            observation.evidence is EvidenceSource.RGB_CABLE
            and observation.confidence
            >= self.config.minimum_rgb_confirmation_confidence
        )

    @classmethod
    def _pending_reason(
        cls,
        observation: HazardObservation,
        can_confirm: bool,
    ) -> CandidateDecisionReason:
        if cls._is_support(observation.evidence):
            return CandidateDecisionReason.SUPPORT_ONLY
        if not can_confirm:
            return CandidateDecisionReason.LOW_CONFIDENCE_RGB
        return CandidateDecisionReason.WAITING_FOR_CONFIRMATION

    @staticmethod
    def _can_fuse_with_track(track: _Track) -> bool:
        support_only = not track.observation_centroids
        confirmable_evidence = {
            EvidenceSource.RGB_CABLE,
            EvidenceSource.STRONG_GEOMETRY,
        }
        return support_only or bool(track.evidence & confirmable_evidence)

    def _observe_refresh(
        self, track: _Track, observation: HazardObservation
    ) -> tuple[ConfirmedHazard, ...]:
        centroid = observation.centroid
        if not self._can_confirm(observation):
            observation_key = (
                observation.sensor_stamp_ns,
                observation.evidence,
                centroid,
            )
            if observation_key in track.observation_keys:
                return ()
            track.observation_keys.add(observation_key)
            track.support_centroids.append(centroid)
            track.last_stamp_ns = max(track.last_stamp_ns, observation.sensor_stamp_ns)
            track.evidence.add(observation.evidence)
            track.confidence = max(track.confidence, observation.confidence)
            return ()
        if observation.sensor_stamp_ns in track.refresh_stamps:
            return ()
        refresh_expired = bool(track.refresh_stamps) and (
            max(*track.refresh_stamps, observation.sensor_stamp_ns)
            - min(*track.refresh_stamps, observation.sensor_stamp_ns)
            > self.config.confirmation_window_ns
        )
        refresh_inconsistent = (
            bool(track.refresh_centroids)
            and self._distance(
                centroid,
                self._centroid(track.refresh_centroids),
            )
            > self.config.association_radius_m
        )
        if refresh_expired or refresh_inconsistent:
            self._clear_refresh(track)
        track.refresh_centroids.append(centroid)
        track.refresh_stamps.add(observation.sensor_stamp_ns)
        track.refresh_last_stamp_ns = max(track.refresh_stamps)
        if (
            track.refresh_latest_points_stamp_ns is None
            or observation.sensor_stamp_ns
            >= track.refresh_latest_points_stamp_ns
        ):
            track.refresh_latest_points = observation.points_odom
            track.refresh_latest_points_stamp_ns = observation.sensor_stamp_ns
        track.refresh_evidence.add(observation.evidence)
        track.refresh_confidence = max(track.refresh_confidence, observation.confidence)
        if len(track.refresh_centroids) < 2:
            return ()
        track.confirmation_latency_ns = (
            max(track.refresh_stamps) - min(track.refresh_stamps)
        )
        track.last_stamp_ns = max(track.last_stamp_ns, *track.refresh_stamps)
        track.observation_centroids.extend(track.refresh_centroids)
        track.confirmation_stamps.update(track.refresh_stamps)
        if (
            track.refresh_latest_points_stamp_ns is not None
            and track.refresh_latest_points_stamp_ns
            >= track.latest_points_stamp_ns
        ):
            track.latest_points = track.refresh_latest_points
            track.latest_points_stamp_ns = (
                track.refresh_latest_points_stamp_ns
            )
        track.evidence.update(track.refresh_evidence)
        track.confidence = max(track.confidence, track.refresh_confidence)
        self._clear_refresh(track)
        return (self._confirmed_hazard(track),)

    @staticmethod
    def _clear_refresh(track: _Track) -> None:
        track.refresh_centroids.clear()
        track.refresh_last_stamp_ns = None
        track.refresh_latest_points = ()
        track.refresh_latest_points_stamp_ns = None
        track.refresh_stamps.clear()
        track.refresh_evidence.clear()
        track.refresh_confidence = 0.0

    @staticmethod
    def _centroid(points: list[Point3]) -> Point3:
        return tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )

    def _expire_at(self, sensor_now_ns: int, *, expire_confirmed: bool = True) -> None:
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
            hazard_track_id=track.hazard_track_id,
            sensor_stamp_ns=track.latest_points_stamp_ns,
            points_odom=track.latest_points,
            centroid=track.centroid,
            observation_count=len(track.observation_centroids),
            evidence=tuple(sorted(track.evidence, key=lambda item: item.value)),
            confidence=self._track_confidence(track),
            spatial_spread_m=spread,
            confirmation_latency_ns=track.confirmation_latency_ns,
        )

    @staticmethod
    def _track_confidence(track: _Track) -> float:
        return min(
            1.0,
            track.confidence
            + (0.05 if EvidenceSource.WEAK_HEIGHT in track.evidence else 0.0)
            + (0.03 if EvidenceSource.INVALID_DEPTH in track.evidence else 0.0),
        )

    @staticmethod
    def _distance(first: Point3, second: Point3) -> float:
        return sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


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


def _slerp(first: Quaternion, second: Quaternion, fraction: float) -> Quaternion:
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
