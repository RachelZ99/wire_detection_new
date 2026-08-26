"""Weak-height and continuous invalid-depth evidence on observed floor."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot, isfinite
from typing import Sequence

from .cable import ObservedFloorRegion
from .geometry import CameraIntrinsics, GroundPlane, Point3
from .temporal import EvidenceSource


Pixel = tuple[int, int]


@dataclass(frozen=True)
class DepthEvidenceConfig:
    sample_stride_px: int = 1
    minimum_weak_height_m: float = 0.006
    ground_noise_multiplier: float = 3.0
    strong_height_m: float = 0.015
    minimum_weak_support_points: int = 8
    cluster_cell_m: float = 0.04
    minimum_physical_span_m: float = 0.04
    minimum_invalid_pixels: int = 12
    minimum_invalid_span_px: float = 12.0
    maximum_invalid_width_px: float = 8.0
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 3.0
    maximum_output_points: int = 160

    def __post_init__(self) -> None:
        if self.sample_stride_px < 1:
            raise ValueError("sample_stride_px must be positive")
        if not 0.0 < self.minimum_weak_height_m < self.strong_height_m:
            raise ValueError("weak height range is invalid")
        if self.ground_noise_multiplier <= 0.0:
            raise ValueError("ground_noise_multiplier must be positive")
        if self.minimum_weak_support_points < 2:
            raise ValueError("minimum_weak_support_points must be at least two")
        if self.cluster_cell_m <= 0.0:
            raise ValueError("cluster_cell_m must be positive")
        if self.minimum_physical_span_m <= 0.0:
            raise ValueError("minimum_physical_span_m must be positive")
        if self.minimum_invalid_pixels < 2:
            raise ValueError("minimum_invalid_pixels must be at least two")
        if not 0.0 < self.minimum_invalid_span_px:
            raise ValueError("minimum_invalid_span_px must be positive")
        if self.maximum_invalid_width_px <= 0.0:
            raise ValueError("maximum_invalid_width_px must be positive")


@dataclass(frozen=True)
class DepthEvidenceCandidate:
    evidence: EvidenceSource
    points_camera: tuple[Point3, ...]
    centroid: Point3
    support_count: int
    physical_span_m: float
    confidence: float


class DepthEvidenceDetector:
    """Extract support-only depth evidence without confirming it directly."""

    def __init__(self, config: DepthEvidenceConfig | None = None) -> None:
        self.config = config or DepthEvidenceConfig()

    def detect(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        floor_region: ObservedFloorRegion,
        *,
        depth_unit_m: float,
        ground_noise_m: float,
    ) -> tuple[DepthEvidenceCandidate, ...]:
        if len(depth_values) < intrinsics.width * intrinsics.height:
            raise ValueError("depth data is smaller than width times height")
        weak = self._weak_height(
            depth_values,
            intrinsics,
            ground,
            depth_unit_m=depth_unit_m,
            ground_noise_m=ground_noise_m,
        )
        invalid = self._invalid_depth(depth_values, intrinsics, ground, floor_region)
        return tuple(
            sorted(
                (*weak, *invalid),
                key=lambda item: (item.evidence.value, item.centroid),
            )
        )

    def _weak_height(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        *,
        depth_unit_m: float,
        ground_noise_m: float,
    ) -> tuple[DepthEvidenceCandidate, ...]:
        threshold = max(
            self.config.minimum_weak_height_m,
            self.config.ground_noise_multiplier * ground_noise_m,
        )
        cells: dict[tuple[int, int], list[Point3]] = {}
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
                if not threshold <= height < self.config.strong_height_m:
                    continue
                floor_point = ground.intersect_pixel_ray(
                    intrinsics, column=column, row=row
                )
                if floor_point is None:
                    continue
                key = (
                    floor(floor_point[0] / self.config.cluster_cell_m),
                    floor(floor_point[2] / self.config.cluster_cell_m),
                )
                cells.setdefault(key, []).append(floor_point)
        candidates = []
        for points in _metric_components(cells):
            candidate = self._candidate(
                EvidenceSource.WEAK_HEIGHT,
                points,
                self.config.minimum_weak_support_points,
                0.58,
            )
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _invalid_depth(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        floor_region: ObservedFloorRegion,
    ) -> tuple[DepthEvidenceCandidate, ...]:
        invalid = {
            (column, row)
            for row in range(intrinsics.height)
            for column in range(intrinsics.width)
            if floor_region.contains(column, row)
            and (
                not isfinite(float(depth_values[row * intrinsics.width + column]))
                or float(depth_values[row * intrinsics.width + column]) <= 0.0
            )
            and self._enclosed_by_valid_depth(depth_values, intrinsics, column, row)
        }
        candidates = []
        for component in _pixel_components(invalid):
            if len(component) < self.config.minimum_invalid_pixels:
                continue
            columns = [pixel[0] for pixel in component]
            rows = [pixel[1] for pixel in component]
            width = max(columns) - min(columns) + 1
            height = max(rows) - min(rows) + 1
            major = float(max(width, height))
            minor = float(min(width, height))
            if (
                major < self.config.minimum_invalid_span_px
                or minor > self.config.maximum_invalid_width_px
            ):
                continue
            points = tuple(
                point
                for column, row in sorted(
                    component, key=lambda item: (item[1], item[0])
                )
                if (
                    point := ground.intersect_pixel_ray(
                        intrinsics, column=column, row=row
                    )
                )
                is not None
                and self.config.minimum_depth_m
                <= point[2]
                <= self.config.maximum_depth_m
            )
            candidate = self._candidate(
                EvidenceSource.INVALID_DEPTH,
                points,
                self.config.minimum_invalid_pixels,
                0.52,
            )
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _enclosed_by_valid_depth(
        self,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        column: int,
        row: int,
    ) -> bool:
        maximum_distance = int(self.config.maximum_invalid_width_px) + 1

        def valid(test_column: int, test_row: int) -> bool:
            if not (
                0 <= test_column < intrinsics.width
                and 0 <= test_row < intrinsics.height
            ):
                return False
            value = float(depth_values[test_row * intrinsics.width + test_column])
            return isfinite(value) and value > 0.0

        return any(
            valid(column - distance, row)
            and valid(column + distance, row)
            or valid(column, row - distance)
            and valid(column, row + distance)
            for distance in range(1, maximum_distance + 1)
        )

    def _candidate(
        self,
        evidence: EvidenceSource,
        points: Sequence[Point3],
        minimum_support: int,
        base_confidence: float,
    ) -> DepthEvidenceCandidate | None:
        if len(points) < minimum_support:
            return None
        span = hypot(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[2] for point in points) - min(point[2] for point in points),
        )
        if span < self.config.minimum_physical_span_m:
            return None
        stride = max(1, len(points) // self.config.maximum_output_points)
        output = tuple(points[::stride][: self.config.maximum_output_points])
        centroid = tuple(
            sum(point[axis] for point in output) / len(output) for axis in range(3)
        )
        return DepthEvidenceCandidate(
            evidence=evidence,
            points_camera=output,
            centroid=centroid,
            support_count=len(points),
            physical_span_m=span,
            confidence=min(0.7, base_confidence + min(0.1, span)),
        )


def _metric_components(
    cells: dict[tuple[int, int], list[Point3]],
) -> tuple[tuple[Point3, ...], ...]:
    remaining = set(cells)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        keys = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for delta_x in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbor = (current[0] + delta_x, current[1] + delta_z)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        keys.append(neighbor)
                        frontier.append(neighbor)
        components.append(tuple(point for key in keys for point in cells[key]))
    return tuple(components)


def _pixel_components(pixels: set[Pixel]) -> tuple[set[Pixel], ...]:
    remaining = set(pixels)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            column, row = frontier.pop()
            for delta_column in (-1, 0, 1):
                for delta_row in (-1, 0, 1):
                    neighbor = (column + delta_column, row + delta_row)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
        components.append(component)
    return tuple(components)
