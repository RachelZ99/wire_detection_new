import math
import unittest

from low_profile_hazard_perception.cable import (
    DiagnosticPinkConfig,
    ObservedFloorRegion,
    TrainingFreeCableConfig,
    TrainingFreeCableDetector,
    diagnostic_pink_pixel_count,
)
from low_profile_hazard_perception.geometry import (
    CameraIntrinsics,
    GroundPlane,
)


def _floor_scene() -> tuple[list[int], CameraIntrinsics, GroundPlane]:
    intrinsics = CameraIntrinsics(
        width=120,
        height=72,
        fx=100.0,
        fy=100.0,
        cx=60.0,
        cy=36.0,
    )
    pitch = math.radians(3.0)
    ground = GroundPlane(
        normal=(0.0, -math.cos(pitch), -math.sin(pitch)),
        offset_m=0.225,
    )
    depth: list[int] = []
    for row in range(intrinsics.height):
        for column in range(intrinsics.width):
            ray = (
                (column - intrinsics.cx) / intrinsics.fx,
                (row - intrinsics.cy) / intrinsics.fy,
                1.0,
            )
            denominator = sum(
                left * right
                for left, right in zip(ground.normal, ray, strict=True)
            )
            depth.append(
                0
                if denominator >= -1e-6
                else int(round(-ground.offset_m / denominator * 1000.0))
            )
    return depth, intrinsics, ground


def _rgb_line(
    intrinsics: CameraIntrinsics,
    *,
    color: tuple[int, int, int],
    width_px: int,
) -> bytes:
    background = (82, 86, 88)
    data = bytearray(background * (intrinsics.width * intrinsics.height))
    first_column = 60 - width_px // 2
    for row in range(48, 69):
        curve_offset = int(round(5.0 * math.sin((row - 48) / 20.0)))
        for column in range(first_column, first_column + width_px):
            offset = (
                (row * intrinsics.width + column + curve_offset) * 3
            )
            data[offset : offset + 3] = bytes(color)
    return bytes(data)


class TrainingFreeRgbCableTests(unittest.TestCase):
    def test_branched_ridge_is_rejected_by_curve_continuity(self) -> None:
        depth, intrinsics, ground = _floor_scene()
        floor = ObservedFloorRegion.from_depth(
            depth,
            intrinsics,
            ground,
            depth_unit_m=0.001,
        )
        data = bytearray((82, 86, 88) * (intrinsics.width * intrinsics.height))

        def paint(center_column: int, row: int) -> None:
            for column in range(center_column - 1, center_column + 2):
                offset = (row * intrinsics.width + column) * 3
                data[offset : offset + 3] = bytes((235, 235, 235))

        for row in range(55, 69):
            paint(60, row)
        for step, row in enumerate(range(55, 47, -1)):
            paint(60 - step, row)
            paint(60 + step, row)

        candidates = TrainingFreeCableDetector(
            TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
                minimum_width_consistency=0.0,
                minimum_curve_continuity=0.80,
            )
        ).detect(bytes(data), intrinsics, ground, floor)

        self.assertEqual(candidates, ())

    def test_inconsistent_width_is_rejected_instead_of_only_downscored(
        self,
    ) -> None:
        depth, intrinsics, ground = _floor_scene()
        floor = ObservedFloorRegion.from_depth(
            depth,
            intrinsics,
            ground,
            depth_unit_m=0.001,
        )
        data = bytearray((82, 86, 88) * (intrinsics.width * intrinsics.height))
        for row in range(48, 69):
            width_px = 2 if row < 58 else 6
            first_column = 60 - width_px // 2
            for column in range(first_column, first_column + width_px):
                offset = (row * intrinsics.width + column) * 3
                data[offset : offset + 3] = bytes((235, 235, 235))

        candidates = TrainingFreeCableDetector(
            TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
                minimum_width_consistency=0.55,
                minimum_curve_continuity=0.80,
            )
        ).detect(bytes(data), intrinsics, ground, floor)

        self.assertEqual(candidates, ())

    def test_color_specific_demo_is_only_a_diagnostic_comparison(self) -> None:
        depth, intrinsics, ground = _floor_scene()
        floor = ObservedFloorRegion.from_depth(
            depth,
            intrinsics,
            ground,
            depth_unit_m=0.001,
        )
        low_contrast_pink = _rgb_line(
            intrinsics,
            color=(100, 80, 90),
            width_px=3,
        )

        diagnostic_count = diagnostic_pink_pixel_count(
            low_contrast_pink,
            intrinsics,
            floor,
            DiagnosticPinkConfig(),
        )
        formal_candidates = TrainingFreeCableDetector().detect(
            low_contrast_pink,
            intrinsics,
            ground,
            floor,
        )

        self.assertGreater(diagnostic_count, 0)
        self.assertEqual(formal_candidates, ())

    def test_floor_constrained_negative_replay_set_stays_empty(self) -> None:
        depth, intrinsics, ground = _floor_scene()
        background = (120, 122, 124)

        def image() -> bytearray:
            return bytearray(
                background * (intrinsics.width * intrinsics.height)
            )

        def stripe(
            data: bytearray,
            columns: range,
            rows: range,
            color: tuple[int, int, int],
        ) -> None:
            for row in rows:
                for column in columns:
                    offset = (row * intrinsics.width + column) * 3
                    data[offset : offset + 3] = bytes(color)

        empty_reflective_floor = image()
        stripe(
            empty_reflective_floor,
            range(44, 76),
            range(52, 69),
            (150, 152, 154),
        )
        long_shadow = image()
        stripe(long_shadow, range(50, 66), range(48, 69), (75, 77, 79))
        cable_reflection = image()
        stripe(cable_reflection, range(60, 61), range(48, 69), (220, 220, 220))
        background_structure = image()
        stripe(
            background_structure,
            range(59, 62),
            range(10, 32),
            (235, 235, 235),
        )
        hanging_wire = image()
        stripe(hanging_wire, range(38, 41), range(48, 69), (235, 235, 235))
        hanging_depth = list(depth)
        for row in range(48, 69):
            for column in range(38, 41):
                index = row * intrinsics.width + column
                hanging_depth[index] = max(1, hanging_depth[index] - 180)
        tripod_leg = image()
        stripe(tripod_leg, range(58, 63), range(48, 69), (50, 50, 50))
        leg_depth = list(depth)
        for row in range(48, 69):
            for column in range(58, 63):
                index = row * intrinsics.width + column
                leg_depth[index] = max(1, leg_depth[index] - 180)
        scenes = {
            "empty reflective floor": (empty_reflective_floor, depth),
            "long shadow": (long_shadow, depth),
            "cable reflection": (cable_reflection, depth),
            "background structure": (background_structure, depth),
            "hanging wire over visible floor": (
                hanging_wire,
                hanging_depth,
            ),
            "table/tripod leg": (tripod_leg, leg_depth),
        }
        detector = TrainingFreeCableDetector(
            TrainingFreeCableConfig(
                local_contrast_threshold=18.0,
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            )
        )

        for name, (rgb, scene_depth) in scenes.items():
            with self.subTest(scene=name):
                floor = ObservedFloorRegion.from_depth(
                    scene_depth,
                    intrinsics,
                    ground,
                    depth_unit_m=0.001,
                )
                self.assertEqual(
                    detector.detect(bytes(rgb), intrinsics, ground, floor),
                    (),
                )

    def test_one_pixel_floor_seam_is_not_a_cable_candidate(self) -> None:
        depth, intrinsics, ground = _floor_scene()
        floor = ObservedFloorRegion.from_depth(
            depth,
            intrinsics,
            ground,
            depth_unit_m=0.001,
        )
        data = bytearray((120, 122, 124) * (intrinsics.width * intrinsics.height))
        for row in range(48, 69):
            offset = (row * intrinsics.width + 60) * 3
            data[offset : offset + 3] = bytes((55, 57, 59))

        candidates = TrainingFreeCableDetector(
            TrainingFreeCableConfig(
                local_contrast_threshold=18.0,
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            )
        ).detect(bytes(data), intrinsics, ground, floor)

        self.assertEqual(candidates, ())

    def test_pale_and_white_two_to_five_pixel_cables_form_floor_candidates(
        self,
    ) -> None:
        depth, intrinsics, ground = _floor_scene()
        # The cable stripe itself has no valid depth. Neighboring observed-floor
        # support must bridge it, and every RGB pixel is positioned by ray-plane
        # intersection rather than borrowing a synchronized depth value.
        for row in range(48, 69):
            for column in range(54, 68):
                depth[row * intrinsics.width + column] = 0
        floor = ObservedFloorRegion.from_depth(
            depth,
            intrinsics,
            ground,
            depth_unit_m=0.001,
        )
        detector = TrainingFreeCableDetector(
            TrainingFreeCableConfig(
                minimum_component_pixels=12,
                minimum_length_px=12.0,
                minimum_physical_span_m=0.04,
            )
        )

        for color, width_px in (((220, 150, 185), 2), ((235, 235, 235), 5)):
            with self.subTest(color=color, width_px=width_px):
                candidates = detector.detect(
                    _rgb_line(
                        intrinsics,
                        color=color,
                        width_px=width_px,
                    ),
                    intrinsics,
                    ground,
                    floor,
                )

                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertGreaterEqual(candidate.pixel_span_px, 12.0)
                self.assertGreaterEqual(candidate.physical_span_m, 0.04)
                self.assertLessEqual(candidate.p90_width_px, 5.5)
                self.assertGreater(candidate.width_consistency, 0.5)
                self.assertGreater(candidate.curve_continuity, 0.8)
                self.assertGreater(candidate.mean_local_contrast, 24.0)
                self.assertGreater(len(candidate.mask_pixels), 12)
                self.assertGreaterEqual(
                    max(
                        sum(
                            pixel_row == row
                            for _, pixel_row in candidate.mask_pixels
                        )
                        for row in range(intrinsics.height)
                    ),
                    width_px,
                )
                self.assertTrue(
                    all(
                        abs(ground.signed_height(point)) < 1e-9
                        for point in candidate.points_camera
                    )
                )


if __name__ == "__main__":
    unittest.main()
