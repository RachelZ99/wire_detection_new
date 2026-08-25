"""Observed-floor geometry for the independent depth evidence path.

The implementation fits inverse depth because a 3-D plane projects to an
affine inverse-depth surface.  The accepted result is still represented and
scored as a metric 3-D plane; inverse depth is only a sparse fitting utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, floor, isfinite, radians, sqrt
from typing import Sequence


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def deproject(self, column: int, row: int, depth_m: float) -> Point3:
        return (
            (column - self.cx) * depth_m / self.fx,
            (row - self.cy) * depth_m / self.fy,
            depth_m,
        )


@dataclass(frozen=True)
class GroundPlane:
    normal: Point3
    offset_m: float

    @property
    def camera_height_m(self) -> float:
        """Perpendicular distance from the optical origin to the floor."""
        return self.offset_m

    def signed_height(self, point: Point3) -> float:
        """Return height above the floor, positive towards the camera."""
        return (
            self.normal[0] * point[0]
            + self.normal[1] * point[1]
            + self.normal[2] * point[2]
            + self.offset_m
        )

    def intersect_pixel_ray(
        self,
        intrinsics: CameraIntrinsics,
        *,
        column: int,
        row: int,
    ) -> Point3 | None:
        """Intersect an optical pixel ray with this observed floor plane."""
        ray = (
            (column - intrinsics.cx) / intrinsics.fx,
            (row - intrinsics.cy) / intrinsics.fy,
            1.0,
        )
        denominator = sum(
            normal * direction
            for normal, direction in zip(self.normal, ray, strict=True)
        )
        if abs(denominator) <= 1e-12:
            return None
        scale = -self.offset_m / denominator
        if not isfinite(scale) or scale <= 0.0:
            return None
        return tuple(scale * direction for direction in ray)


@dataclass(frozen=True)
class GroundQualityMetrics:
    support_count: int
    sampled_valid_count: int
    inlier_ratio: float
    median_residual_m: float
    p90_residual_m: float
    spatial_coverage: float
    temporal_consistency: float
    normal_change_degrees: float
    height_change_m: float
    nominal_height_error_m: float


@dataclass(frozen=True)
class GroundEstimate:
    accepted: bool
    model: GroundPlane
    metrics: GroundQualityMetrics
    reason: str = ""


@dataclass(frozen=True)
class GroundEstimatorConfig:
    sample_stride_px: int = 6
    ransac_iterations: int = 160
    ransac_score_max_samples: int = 1200
    inlier_threshold_m: float = 0.008
    minimum_support: int = 500
    minimum_inlier_ratio: float = 0.70
    maximum_p90_residual_m: float = 0.008
    minimum_spatial_coverage: float = 0.35
    maximum_ground_tilt_degrees: float = 25.0
    temporal_angle_tolerance_degrees: float = 4.0
    temporal_height_tolerance_m: float = 0.025
    minimum_temporal_consistency: float = 0.20
    temporal_smoothing_factor: float = 0.35
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 4.0

    def __post_init__(self) -> None:
        if self.sample_stride_px < 1:
            raise ValueError("sample_stride_px must be positive")
        if self.ransac_iterations < 1:
            raise ValueError("ransac_iterations must be positive")
        if self.ransac_score_max_samples < 3:
            raise ValueError("ransac_score_max_samples must be at least three")
        if self.minimum_support < 3:
            raise ValueError("minimum_support must be at least three")
        if not 0.0 < self.temporal_smoothing_factor <= 1.0:
            raise ValueError("temporal_smoothing_factor must be in (0, 1]")
        if not 0.0 <= self.minimum_temporal_consistency <= 1.0:
            raise ValueError("minimum_temporal_consistency must be in [0, 1]")


@dataclass(frozen=True)
class _Sample:
    column: int
    row: int
    normalized_x: float
    normalized_y: float
    depth_m: float

    @property
    def inverse_depth(self) -> float:
        return 1.0 / self.depth_m


class GroundEstimator:
    """Deterministically fit and quality-gate one observed floor per frame."""

    def __init__(self, config: GroundEstimatorConfig | None = None) -> None:
        self.config = config or GroundEstimatorConfig()
        self._previous: GroundPlane | None = None

    def estimate(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        *,
        depth_unit_m: float,
        nominal_camera_height_m: float,
    ) -> GroundEstimate:
        if len(depth_values) < intrinsics.width * intrinsics.height:
            raise ValueError("depth data is smaller than width times height")
        samples = self._samples(depth_values, intrinsics, depth_unit_m)
        empty_plane = GroundPlane((0.0, -1.0, 0.0), 0.0)
        if len(samples) < self.config.minimum_support:
            return GroundEstimate(
                accepted=False,
                model=empty_plane,
                metrics=self._empty_metrics(
                    len(samples), nominal_camera_height_m
                ),
                reason="insufficient valid floor samples",
            )

        coefficients = self._ransac(samples)
        if coefficients is None:
            return GroundEstimate(
                accepted=False,
                model=empty_plane,
                metrics=self._empty_metrics(
                    len(samples), nominal_camera_height_m
                ),
                reason="no direction-consistent floor plane",
            )

        for _ in range(2):
            plane = _plane_from_inverse_depth(coefficients)
            inliers = [
                sample
                for sample in samples
                if abs(plane.signed_height(_point(sample)))
                <= self.config.inlier_threshold_m
            ]
            refined = _least_squares_inverse_depth(inliers)
            if refined is None:
                break
            coefficients = refined

        model = _plane_from_inverse_depth(coefficients)
        residual_samples = [
            (abs(model.signed_height(_point(sample))), sample)
            for sample in samples
        ]
        inlier_pairs = [
            pair
            for pair in residual_samples
            if pair[0] <= self.config.inlier_threshold_m
        ]
        residuals = sorted(pair[0] for pair in inlier_pairs)
        support = len(inlier_pairs)
        inlier_ratio = support / len(samples)
        coverage = _spatial_coverage(
            [pair[1] for pair in inlier_pairs], intrinsics
        )
        median = _percentile(residuals, 0.50)
        p90 = _percentile(residuals, 0.90)
        normal_change, height_change, consistency = self._consistency(model)
        metrics = GroundQualityMetrics(
            support_count=support,
            sampled_valid_count=len(samples),
            inlier_ratio=inlier_ratio,
            median_residual_m=median,
            p90_residual_m=p90,
            spatial_coverage=coverage,
            temporal_consistency=consistency,
            normal_change_degrees=normal_change,
            height_change_m=height_change,
            nominal_height_error_m=abs(
                model.camera_height_m - nominal_camera_height_m
            ),
        )
        reason = self._rejection_reason(metrics)
        accepted = not reason
        if accepted:
            if self._previous is not None:
                model = _blend_planes(
                    self._previous,
                    model,
                    self.config.temporal_smoothing_factor,
                )
            self._previous = model
        return GroundEstimate(accepted, model, metrics, reason)

    def _samples(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        depth_unit_m: float,
    ) -> list[_Sample]:
        samples: list[_Sample] = []
        stride = self.config.sample_stride_px
        for row in range(0, intrinsics.height, stride):
            offset = row * intrinsics.width
            normalized_y = (row - intrinsics.cy) / intrinsics.fy
            for column in range(0, intrinsics.width, stride):
                depth_m = float(depth_values[offset + column]) * depth_unit_m
                if (
                    not isfinite(depth_m)
                    or depth_m < self.config.minimum_depth_m
                    or depth_m > self.config.maximum_depth_m
                ):
                    continue
                samples.append(
                    _Sample(
                        column,
                        row,
                        (column - intrinsics.cx) / intrinsics.fx,
                        normalized_y,
                        depth_m,
                    )
                )
        return samples

    def _ransac(
        self, samples: Sequence[_Sample]
    ) -> tuple[float, float, float] | None:
        best: tuple[int, float, tuple[float, float, float]] | None = None
        count = len(samples)
        score_stride = max(1, count // self.config.ransac_score_max_samples)
        score_samples = samples[::score_stride]
        state = 0x5EED1234
        minimum_up_dot = cos(radians(self.config.maximum_ground_tilt_degrees))
        for _ in range(self.config.ransac_iterations):
            indices: list[int] = []
            while len(indices) < 3:
                state = (1664525 * state + 1013904223) & 0xFFFFFFFF
                index = state % count
                if index not in indices:
                    indices.append(index)
            coefficients = _inverse_depth_through(
                samples[indices[0]], samples[indices[1]], samples[indices[2]]
            )
            if coefficients is None:
                continue
            plane = _plane_from_inverse_depth(coefficients)
            if -plane.normal[1] < minimum_up_dot:
                continue
            residuals = [
                abs(plane.signed_height(_point(sample)))
                for sample in score_samples
            ]
            inliers = [
                value
                for value in residuals
                if value <= self.config.inlier_threshold_m
            ]
            if not inliers:
                continue
            score = (len(inliers), -_percentile(sorted(inliers), 0.90))
            candidate = (score[0], score[1], coefficients)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return None if best is None else best[2]

    def _consistency(self, model: GroundPlane) -> tuple[float, float, float]:
        if self._previous is None:
            return 0.0, 0.0, 1.0
        dot = max(
            -1.0,
            min(
                1.0,
                sum(
                    left * right
                    for left, right in zip(
                        self._previous.normal, model.normal, strict=True
                    )
                ),
            ),
        )
        angle = acos(dot) * 180.0 / 3.141592653589793
        height = abs(self._previous.camera_height_m - model.camera_height_m)
        angle_score = max(
            0.0, 1.0 - angle / self.config.temporal_angle_tolerance_degrees
        )
        height_score = max(
            0.0, 1.0 - height / self.config.temporal_height_tolerance_m
        )
        return angle, height, min(angle_score, height_score)

    def _rejection_reason(self, metrics: GroundQualityMetrics) -> str:
        if metrics.support_count < self.config.minimum_support:
            return "insufficient ground support"
        if metrics.inlier_ratio < self.config.minimum_inlier_ratio:
            return "ground inlier ratio is too low"
        if metrics.p90_residual_m > self.config.maximum_p90_residual_m:
            return "ground residual is too large"
        if metrics.spatial_coverage < self.config.minimum_spatial_coverage:
            return "ground spatial coverage is too small"
        if (
            metrics.temporal_consistency
            < self.config.minimum_temporal_consistency
        ):
            return "ground temporal consistency is too low"
        return ""

    @staticmethod
    def _empty_metrics(
        sampled_valid_count: int, nominal_camera_height_m: float
    ) -> GroundQualityMetrics:
        return GroundQualityMetrics(
            support_count=0,
            sampled_valid_count=sampled_valid_count,
            inlier_ratio=0.0,
            median_residual_m=0.0,
            p90_residual_m=0.0,
            spatial_coverage=0.0,
            temporal_consistency=0.0,
            normal_change_degrees=0.0,
            height_change_m=0.0,
            nominal_height_error_m=abs(nominal_camera_height_m),
        )


@dataclass(frozen=True)
class StrongGeometryConfig:
    """Profile parameters for class-agnostic raised geometry."""

    sample_stride_px: int = 2
    strong_height_m: float = 0.015
    maximum_height_m: float = 0.150
    minimum_support_points: int = 18
    cluster_cell_m: float = 0.04
    minimum_spatial_span_m: float = 0.04
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 3.0

    def __post_init__(self) -> None:
        if self.sample_stride_px < 1:
            raise ValueError("sample_stride_px must be positive")
        if self.minimum_support_points < 2:
            raise ValueError("minimum_support_points must be at least two")
        if not 0.0 < self.strong_height_m < self.maximum_height_m:
            raise ValueError("strong height range is invalid")
        if self.cluster_cell_m <= 0.0:
            raise ValueError("cluster_cell_m must be positive")


@dataclass(frozen=True)
class GeometricCandidate:
    """One locally supported protrusion expressed in the camera frame."""

    points: tuple[Point3, ...]
    centroid: Point3
    support_count: int
    p20_height_m: float
    p90_height_m: float
    spatial_span_m: float
    confidence: float


class StrongGeometryDetector:
    """Cluster strong signed-height points in metric bird's-eye cells."""

    def __init__(self, config: StrongGeometryConfig | None = None) -> None:
        self.config = config or StrongGeometryConfig()

    def detect(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        *,
        depth_unit_m: float,
    ) -> tuple[GeometricCandidate, ...]:
        if len(depth_values) < intrinsics.width * intrinsics.height:
            raise ValueError("depth data is smaller than width times height")
        cells: dict[tuple[int, int], list[tuple[Point3, float]]] = {}
        stride = self.config.sample_stride_px
        for row in range(0, intrinsics.height, stride):
            offset = row * intrinsics.width
            for column in range(0, intrinsics.width, stride):
                depth_m = float(depth_values[offset + column]) * depth_unit_m
                if (
                    not isfinite(depth_m)
                    or depth_m < self.config.minimum_depth_m
                    or depth_m > self.config.maximum_depth_m
                ):
                    continue
                point = intrinsics.deproject(column, row, depth_m)
                height = ground.signed_height(point)
                if not (
                    self.config.strong_height_m
                    <= height
                    <= self.config.maximum_height_m
                ):
                    continue
                key = (
                    floor(point[0] / self.config.cluster_cell_m),
                    floor(point[2] / self.config.cluster_cell_m),
                )
                cells.setdefault(key, []).append((point, height))

        candidates: list[GeometricCandidate] = []
        remaining = set(cells)
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            frontier = [seed]
            cluster_keys: list[tuple[int, int]] = []
            while frontier:
                current = frontier.pop()
                cluster_keys.append(current)
                for delta_x in (-1, 0, 1):
                    for delta_z in (-1, 0, 1):
                        neighbor = (
                            current[0] + delta_x,
                            current[1] + delta_z,
                        )
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            frontier.append(neighbor)
            supported = [item for key in cluster_keys for item in cells[key]]
            candidate = self._candidate(supported)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-item.confidence, item.centroid),
            )
        )

    def _candidate(
        self, supported: Sequence[tuple[Point3, float]]
    ) -> GeometricCandidate | None:
        if len(supported) < self.config.minimum_support_points:
            return None
        heights = sorted(item[1] for item in supported)
        p20 = _percentile(heights, 0.20)
        p90 = _percentile(heights, 0.90)
        # Requiring even the lower robust quantile to clear the strong gate
        # prevents a few maximum-depth errors from carrying a floor patch.
        if p20 < self.config.strong_height_m:
            return None
        points = tuple(item[0] for item in supported)
        minimum_x = min(point[0] for point in points)
        maximum_x = max(point[0] for point in points)
        minimum_z = min(point[2] for point in points)
        maximum_z = max(point[2] for point in points)
        span = sqrt(
            (maximum_x - minimum_x) ** 2 + (maximum_z - minimum_z) ** 2
        )
        if span < self.config.minimum_spatial_span_m:
            return None
        centroid = tuple(
            sum(point[axis] for point in points) / len(points)
            for axis in range(3)
        )
        support_score = min(
            1.0,
            len(points) / float(self.config.minimum_support_points * 3),
        )
        height_score = min(
            1.0,
            (p20 - self.config.strong_height_m)
            / max(self.config.strong_height_m, 1e-6),
        )
        return GeometricCandidate(
            points=points,
            centroid=centroid,
            support_count=len(points),
            p20_height_m=p20,
            p90_height_m=p90,
            spatial_span_m=span,
            confidence=0.5 + 0.25 * support_score + 0.25 * height_score,
        )


def _point(sample: _Sample) -> Point3:
    return (
        sample.normalized_x * sample.depth_m,
        sample.normalized_y * sample.depth_m,
        sample.depth_m,
    )


def _plane_from_inverse_depth(
    coefficients: tuple[float, float, float],
) -> GroundPlane:
    a, b, c = coefficients
    norm = sqrt(a * a + b * b + c * c)
    if norm <= 1e-12:
        return GroundPlane((0.0, -1.0, 0.0), 0.0)
    # a*x + b*y + c*z - 1 = 0.  Orient the plane so the camera
    # origin has positive signed height above the floor.
    return GroundPlane((-a / norm, -b / norm, -c / norm), 1.0 / norm)


def _blend_planes(
    previous: GroundPlane, current: GroundPlane, factor: float
) -> GroundPlane:
    inverse = 1.0 - factor
    normal = tuple(
        inverse * previous.normal[index] + factor * current.normal[index]
        for index in range(3)
    )
    norm = sqrt(sum(value * value for value in normal))
    return GroundPlane(
        normal=tuple(value / norm for value in normal),
        offset_m=(inverse * previous.offset_m + factor * current.offset_m)
        / norm,
    )


def _inverse_depth_through(
    first: _Sample, second: _Sample, third: _Sample
) -> tuple[float, float, float] | None:
    return _solve_three_by_three(
        (
            (first.normalized_x, first.normalized_y, 1.0),
            (second.normalized_x, second.normalized_y, 1.0),
            (third.normalized_x, third.normalized_y, 1.0),
        ),
        (first.inverse_depth, second.inverse_depth, third.inverse_depth),
    )


def _least_squares_inverse_depth(
    samples: Sequence[_Sample],
) -> tuple[float, float, float] | None:
    if len(samples) < 3:
        return None
    xx = xy = x1 = yy = y1 = 0.0
    xr = yr = one_r = 0.0
    for sample in samples:
        x = sample.normalized_x
        y = sample.normalized_y
        value = sample.inverse_depth
        xx += x * x
        xy += x * y
        x1 += x
        yy += y * y
        y1 += y
        xr += x * value
        yr += y * value
        one_r += value
    return _solve_three_by_three(
        ((xx, xy, x1), (xy, yy, y1), (x1, y1, float(len(samples)))),
        (xr, yr, one_r),
    )


def _solve_three_by_three(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    rows = [list(matrix[index]) + [vector[index]] for index in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(3):
            if row == column:
                continue
            factor = rows[row][column]
            rows[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    rows[row], rows[column], strict=True
                )
            ]
    return rows[0][3], rows[1][3], rows[2][3]


def _spatial_coverage(
    samples: Sequence[_Sample], intrinsics: CameraIntrinsics
) -> float:
    columns = 8
    rows = 6
    occupied = {
        (
            min(columns - 1, sample.column * columns // intrinsics.width),
            min(rows - 1, sample.row * rows // intrinsics.height),
        )
        for sample in samples
    }
    return len(occupied) / float(columns * rows)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * fraction))
    return float(sorted_values[index])
