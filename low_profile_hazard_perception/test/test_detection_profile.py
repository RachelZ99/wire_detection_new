import json
import tempfile
import unittest
from pathlib import Path

from low_profile_hazard_perception.detection_profile import (
    DetectionProfile,
    ProfileBindingState,
    ProfileMismatch,
    default_detection_profile_path,
)


class DetectionProfileBindingTests(unittest.TestCase):
    def test_independent_mismatch_causes_clear_independently(self) -> None:
        binding = ProfileBindingState()
        binding.set("profile:speed_exceeded", active=True)
        binding.set("profile:color_image_mismatch", active=True)

        binding.set("profile:color_image_mismatch", active=False)

        self.assertTrue(binding.mismatched)
        self.assertEqual(binding.reasons, ("profile:speed_exceeded",))
        self.assertEqual(binding.blocking_reason(), "profile:speed_exceeded")

    def test_repository_profile_is_the_formal_initial_envelope(self) -> None:
        profile = DetectionProfile.load(default_detection_profile_path())

        self.assertEqual(profile.profile_id, "dcw2-home-640x360-v1")
        self.assertEqual(profile.camera.image_profile, "640x360@10Hz")
        self.assertEqual(profile.validated_height_range_m, (0.20, 0.25))
        self.assertEqual(profile.maximum_speed_mps, 0.3)
        self.assertEqual(
            profile.resource_budget.depth_geometry_average_cpu_cores,
            1.0,
        )

    def _profile(self) -> DetectionProfile:
        document = {
            "schema_version": 1,
            "profile_id": "dcw2-home-640x360-v1",
            "validation": {
                "phase": "home_feasibility",
                "maximum_speed_mps": 0.3,
            },
            "camera": {
                "width": 640,
                "height": 360,
                "rgb_encoding": "rgb8",
                "depth_encoding": "16UC1",
                "rate_hz": 10.0,
                "validated_rate_range_hz": [8.0, 12.0],
                "frame_id": "camera_1_color_optical_frame",
            },
            "installation": {
                "observed_camera_height_m": 0.225,
                "validated_height_range_m": [0.20, 0.25],
                "observed_downward_pitch_degrees": 2.75,
                "validated_downward_pitch_range_degrees": [1.5, 4.0],
                "footprint_m": {
                    "minimum_x": -0.25,
                    "maximum_x": 0.375,
                    "minimum_y": -0.30,
                    "maximum_y": 0.30,
                },
            },
            "implementation": {
                "rule_version": "training-free-thin-line-v1",
                "model_version": "none",
            },
            "parameters": {
                "expected_width": 640,
                "expected_height": 360,
                "camera_frame": "camera_1_color_optical_frame",
                "operating_speed_mps": 0.3,
                "confirmed_retention_ms": 2000.0,
                "rgb_reorder_capacity": 16,
            },
            "resource_budget": {
                "processing_p95_ms": 80.0,
                "depth_geometry_average_cpu_cores": 1.0,
                "soak_duration_seconds": 7200,
                "maximum_input_queue_depth": 1,
                "maximum_rgb_reorder_depth": 16,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return DetectionProfile.load(path)

    def test_initial_profile_records_the_validated_envelope(self) -> None:
        profile = self._profile()

        self.assertEqual(profile.profile_id, "dcw2-home-640x360-v1")
        self.assertEqual(profile.schema_version, 1)
        self.assertEqual(profile.camera.image_profile, "640x360@10Hz")
        self.assertEqual(profile.maximum_speed_mps, 0.3)
        self.assertEqual(profile.validated_height_range_m, (0.20, 0.25))
        self.assertEqual(profile.model_version, "none")
        self.assertEqual(profile.parameters["confirmed_retention_ms"], 2000.0)
        self.assertEqual(profile.resource_budget.soak_duration_seconds, 7200)
        self.assertEqual(len(profile.fingerprint), 64)

    def test_changed_stream_installation_or_speed_cannot_reuse_profile(self) -> None:
        profile = self._profile()

        with self.assertRaisesRegex(ProfileMismatch, "image profile"):
            profile.validate_image(
                width=640,
                height=400,
                encoding="rgb8",
                is_depth=False,
            )
        with self.assertRaisesRegex(ProfileMismatch, "camera installation"):
            profile.validate_observed_camera_height(0.60)
        with self.assertRaisesRegex(ProfileMismatch, "speed"):
            profile.validate_speed(0.31)
        with self.assertRaisesRegex(ProfileMismatch, "image rate"):
            profile.validate_rate(30.0)
        with self.assertRaisesRegex(ProfileMismatch, "camera installation"):
            profile.validate_observed_downward_pitch(10.0)

    def test_runtime_parameters_are_bound_to_the_versioned_profile(self) -> None:
        profile = self._profile()
        profile.validate_parameters(
            {
                "expected_width": 640,
                "expected_height": 360,
                "camera_frame": "camera_1_color_optical_frame",
                "operating_speed_mps": 0.3,
                "confirmed_retention_ms": 2000.0,
                "rgb_reorder_capacity": 16,
            }
        )

        with self.assertRaisesRegex(ProfileMismatch, "confirmed_retention_ms"):
            profile.validate_parameters(
                {
                    "expected_width": 640,
                    "expected_height": 360,
                    "camera_frame": "camera_1_color_optical_frame",
                    "operating_speed_mps": 0.3,
                    "confirmed_retention_ms": 1500.0,
                    "rgb_reorder_capacity": 16,
                }
            )


if __name__ == "__main__":
    unittest.main()
