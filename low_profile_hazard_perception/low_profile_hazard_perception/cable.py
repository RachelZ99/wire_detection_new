"""Training-free RGB cable evidence constrained by an observed floor."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

from .geometry import CameraIntrinsics, GroundPlane, Point3


Pixel = tuple[int, int]
Rgb = tuple[int, int, int]


@dataclass(frozen=True)
class ObservedFloorRegion:
    """Coarse conservative support map derived from accepted floor depth."""

    width: int
    height: int
    cell_size_px: int
    supported_cells: frozenset[tuple[int, int]]

    @classmethod
    def from_depth(
        cls,
        depth_values: Sequence[int | float],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        *,
        depth_unit_m: float,
        cell_size_px: int = 8,
        sample_stride_px: int = 2,
        floor_tolerance_m: float = 0.012,
        blocking_height_m: float = 0.030,
    ) -> "ObservedFloorRegion":
        if len(depth_values) < intrinsics.width * intrinsics.height:
            raise ValueError("depth data is smaller than width times height")
        if cell_size_px < 2 or sample_stride_px < 1:
            raise ValueError("floor-region sampling bounds must be positive")
        floor_counts: dict[tuple[int, int], int] = {}
        blocked_counts: dict[tuple[int, int], int] = {}
        for row in range(0, intrinsics.height, sample_stride_px):
            offset = row * intrinsics.width
            for column in range(0, intrinsics.width, sample_stride_px):
                depth_m = float(depth_values[offset + column]) * depth_unit_m
                if not isfinite(depth_m) or depth_m <= 0.0:
                    continue
                point = intrinsics.deproject(column, row, depth_m)
                height = ground.signed_height(point)
                cell = (column // cell_size_px, row // cell_size_px)
                if abs(height) <= floor_tolerance_m:
                    floor_counts[cell] = floor_counts.get(cell, 0) + 1
                elif height >= blocking_height_m:
                    blocked_counts[cell] = blocked_counts.get(cell, 0) + 1

        directly_supported = {
            cell
            for cell, count in floor_counts.items()
            if count >= 2 and blocked_counts.get(cell, 0) < 2
        }
        # Bridge only a one-cell hole enclosed by observed floor on opposing
        # sides. This preserves a narrow invalid-depth cable stripe without
        # growing the ROI into unobserved background or beside hanging objects.
        supported: set[tuple[int, int]] = set(directly_supported)
        bridge_candidates = {
            (column + delta_column, row + delta_row)
            for column, row in directly_supported
            for delta_column in (-1, 0, 1)
            for delta_row in (-1, 0, 1)
        }
        opposing_pairs = (
            ((-1, 0), (1, 0)),
            ((0, -1), (0, 1)),
            ((-1, -1), (1, 1)),
            ((-1, 1), (1, -1)),
        )
        for column, row in bridge_candidates:
            if blocked_counts.get((column, row), 0) >= 2:
                continue
            if any(
                (column + first[0], row + first[1]) in directly_supported
                and (column + second[0], row + second[1])
                in directly_supported
                for first, second in opposing_pairs
            ):
                supported.add((column, row))
        return cls(
            width=intrinsics.width,
            height=intrinsics.height,
            cell_size_px=cell_size_px,
            supported_cells=frozenset(supported),
        )

    def contains(self, column: int, row: int) -> bool:
        return (
            0 <= column < self.width
            and 0 <= row < self.height
            and (
                column // self.cell_size_px,
                row // self.cell_size_px,
            )
            in self.supported_cells
        )


@dataclass(frozen=True)
class DiagnosticPinkConfig:
    """Legacy-style color bounds for comparison metrics only."""

    minimum_red: int = 90
    minimum_red_over_green: int = 10
    minimum_blue_over_green: int = 5
    maximum_red_blue_difference: int = 60


def diagnostic_pink_pixel_count(
    rgb_values: bytes | bytearray | memoryview,
    intrinsics: CameraIntrinsics,
    floor_region: ObservedFloorRegion,
    config: DiagnosticPinkConfig | None = None,
) -> int:
    """Count demo-profile pixels without producing operational evidence."""
    expected_size = intrinsics.width * intrinsics.height * 3
    if len(rgb_values) < expected_size:
        raise ValueError("RGB data is smaller than width times height")
    profile = config or DiagnosticPinkConfig()
    values = memoryview(rgb_values)
    count = 0
    for row in range(intrinsics.height):
        for column in range(intrinsics.width):
            if not floor_region.contains(column, row):
                continue
            red, green, blue = _rgb(values, intrinsics.width, column, row)
            count += int(
                red >= profile.minimum_red
                and red - green >= profile.minimum_red_over_green
                and blue - green >= profile.minimum_blue_over_green
                and abs(red - blue)
                <= profile.maximum_red_blue_difference
            )
    return count


@dataclass(frozen=True)
class TrainingFreeCableConfig:
    """Detection-profile parameters; none encode a cable color."""

    local_contrast_threshold: float = 24.0
    maximum_half_width_px: int = 3
    minimum_component_pixels: int = 16
    minimum_length_px: float = 16.0
    minimum_apparent_width_px: float = 1.5
    maximum_apparent_width_px: float = 6.0
    minimum_width_consistency: float = 0.55
    minimum_curve_continuity: float = 0.80
    minimum_physical_span_m: float = 0.06
    maximum_ground_age_ns: int = 500_000_000
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 3.0
    maximum_output_points: int = 160

    def __post_init__(self) -> None:
        if self.local_contrast_threshold <= 0.0:
            raise ValueError("local_contrast_threshold must be positive")
        if self.maximum_half_width_px < 1:
            raise ValueError("maximum_half_width_px must be positive")
        if self.minimum_component_pixels < 2:
            raise ValueError("minimum_component_pixels must be at least two")
        if self.minimum_length_px <= 0.0:
            raise ValueError("minimum_length_px must be positive")
        if not 0.0 < self.minimum_apparent_width_px < self.maximum_apparent_width_px:
            raise ValueError("apparent cable width range is invalid")
        if not 0.0 <= self.minimum_width_consistency <= 1.0:
            raise ValueError("minimum_width_consistency must be in [0, 1]")
        if not 0.0 <= self.minimum_curve_continuity <= 1.0:
            raise ValueError("minimum_curve_continuity must be in [0, 1]")
        if self.minimum_physical_span_m <= 0.0:
            raise ValueError("minimum_physical_span_m must be positive")
        if self.maximum_ground_age_ns <= 0:
            raise ValueError("maximum_ground_age_ns must be positive")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("cable projection depth range is invalid")
        if self.maximum_output_points < 2:
            raise ValueError("maximum_output_points must be at least two")


@dataclass(frozen=True)
class CableCandidate:
    """Floor-projected thin RGB structure in the optical camera frame."""

    mask_pixels: tuple[Pixel, ...]
    points_camera: tuple[Point3, ...]
    centroid: Point3
    mean_local_contrast: float
    pixel_span_px: float
    p90_width_px: float
    width_consistency: float
    curve_continuity: float
    physical_span_m: float
    confidence: float


class TrainingFreeCableDetector:
    """Find locally contrasting paired-edge ridges without a color model."""

    _NORMALS = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, config: TrainingFreeCableConfig | None = None) -> None:
        self.config = config or TrainingFreeCableConfig()

    def detect(
        self,
        rgb_values: bytes | bytearray | memoryview,
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
        floor_region: ObservedFloorRegion,
    ) -> tuple[CableCandidate, ...]:
        expected_size = intrinsics.width * intrinsics.height * 3
        if len(rgb_values) < expected_size:
            raise ValueError("RGB data is smaller than width times height")
        if (
            floor_region.width != intrinsics.width
            or floor_region.height != intrinsics.height
        ):
            raise ValueError("floor region and RGB profile differ")
        values = memoryview(rgb_values)
        margin = self.config.maximum_half_width_px + 1
        responses: dict[Pixel, tuple[float, float]] = {}
        for row in range(margin, intrinsics.height - margin):
            for column in range(margin, intrinsics.width - margin):
                if not floor_region.contains(column, row):
                    continue
                response, width = self._thin_line_response(
                    values, intrinsics.width, column, row
                )
                if response >= self.config.local_contrast_threshold:
                    responses[(column, row)] = (response, width)

        candidates = [
            candidate
            for component in _components(set(responses))
            if (
                candidate := self._candidate(
                    component,
                    responses,
                    intrinsics,
                    ground,
                )
            )
            is not None
        ]
        return tuple(
            sorted(candidates, key=lambda item: (-item.confidence, item.centroid))
        )

    def _thin_line_response(
        self,
        values: memoryview,
        image_width: int,
        column: int,
        row: int,
    ) -> tuple[float, float]:
        center = _rgb(values, image_width, column, row)
        best = (0.0, 0.0)
        for normal_column, normal_row in self._NORMALS:
            for half_width in range(1, self.config.maximum_half_width_px + 1):
                side_distance = half_width + 1
                left = _rgb(
                    values,
                    image_width,
                    column - normal_column * side_distance,
                    row - normal_row * side_distance,
                )
                right = _rgb(
                    values,
                    image_width,
                    column + normal_column * side_distance,
                    row + normal_row * side_distance,
                )
                paired_contrast = min(
                    _color_distance(center, left),
                    _color_distance(center, right),
                )
                side_asymmetry = _color_distance(left, right)
                response = paired_contrast - 0.55 * side_asymmetry
                if response > best[0]:
                    best = (response, float(half_width * 2))
        return best

    def _candidate(
        self,
        component: set[Pixel],
        responses: dict[Pixel, tuple[float, float]],
        intrinsics: CameraIntrinsics,
        ground: GroundPlane,
    ) -> CableCandidate | None:
        if len(component) < self.config.minimum_component_pixels:
            return None
        columns = [pixel[0] for pixel in component]
        rows = [pixel[1] for pixel in component]
        pixel_span = hypot(max(columns) - min(columns), max(rows) - min(rows))
        if pixel_span < self.config.minimum_length_px:
            return None
        apparent_width = len(component) / max(pixel_span, 1.0)
        if not (
            self.config.minimum_apparent_width_px
            <= apparent_width
            <= self.config.maximum_apparent_width_px
        ):
            return None
        widths = sorted(responses[pixel][1] for pixel in component)
        p10_width = _percentile(widths, 0.10)
        p90_width = _percentile(widths, 0.90)
        width_consistency = max(
            0.0,
            1.0
            - (p90_width - p10_width)
            / max(float(self.config.maximum_half_width_px * 2), 1.0),
        )
        continuity = _curve_continuity(component)
        if (
            width_consistency < self.config.minimum_width_consistency
            or continuity < self.config.minimum_curve_continuity
        ):
            return None
        ordered_pixels = tuple(sorted(component, key=lambda pixel: (pixel[1], pixel[0])))
        projected = [
            (pixel, ground.intersect_pixel_ray(
                intrinsics,
                column=pixel[0],
                row=pixel[1],
            ))
            for pixel in ordered_pixels
        ]
        projected = [
            (pixel, point)
            for pixel, point in projected
            if point is not None
            and self.config.minimum_depth_m <= point[2] <= self.config.maximum_depth_m
        ]
        if len(projected) < self.config.minimum_component_pixels:
            return None
        points = [point for _, point in projected]
        physical_span = hypot(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[2] for point in points) - min(point[2] for point in points),
        )
        if physical_span < self.config.minimum_physical_span_m:
            return None
        stride = max(1, len(points) // self.config.maximum_output_points)
        output_points = tuple(points[::stride][: self.config.maximum_output_points])
        centroid = tuple(
            sum(point[axis] for point in output_points) / len(output_points)
            for axis in range(3)
        )
        mean_contrast = sum(responses[pixel][0] for pixel in component) / len(component)
        contrast_score = min(
            1.0,
            mean_contrast / (self.config.local_contrast_threshold * 3.0),
        )
        span_score = min(
            1.0,
            physical_span / (self.config.minimum_physical_span_m * 2.0),
        )
        return CableCandidate(
            mask_pixels=tuple(pixel for pixel, _ in projected),
            points_camera=output_points,
            centroid=centroid,
            mean_local_contrast=mean_contrast,
            pixel_span_px=pixel_span,
            p90_width_px=min(p90_width, apparent_width),
            width_consistency=width_consistency,
            curve_continuity=continuity,
            physical_span_m=physical_span,
            confidence=(
                0.30 * contrast_score
                + 0.25 * span_score
                + 0.25 * width_consistency
                + 0.20 * continuity
            ),
        )


def _rgb(
    values: memoryview,
    image_width: int,
    column: int,
    row: int,
) -> Rgb:
    offset = (row * image_width + column) * 3
    return values[offset], values[offset + 1], values[offset + 2]


def _color_distance(first: Rgb, second: Rgb) -> float:
    return sum(abs(left - right) for left, right in zip(first, second, strict=True)) / 3.0


def _components(pixels: set[Pixel]) -> tuple[set[Pixel], ...]:
    remaining = set(pixels)
    components: list[set[Pixel]] = []
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


def _curve_continuity(component: set[Pixel]) -> float:
    """Score whether a component follows one thin trajectory, not branches.

    Slice along the component's longer image axis and count disjoint runs in
    each slice.  A normal straight or gently curving cable contributes one run
    per slice.  Forks, paired seams, and broad reflections contribute multiple
    runs and therefore reduce the score.
    """
    if not component:
        return 0.0
    columns = [column for column, _ in component]
    rows = [row for _, row in component]
    use_rows = max(rows) - min(rows) >= max(columns) - min(columns)
    slices: dict[int, set[int]] = {}
    for column, row in component:
        major, minor = (row, column) if use_rows else (column, row)
        slices.setdefault(major, set()).add(minor)
    run_count = 0
    for values in slices.values():
        ordered = sorted(values)
        run_count += 1 + sum(
            right - left > 1
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
    return len(slices) / max(run_count, 1)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * fraction))
    return float(sorted_values[index])
