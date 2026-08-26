"""Versioned runtime binding for one validated detection envelope."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProfileMismatch(ValueError):
    """The runtime is outside the envelope validated by a profile."""


class ProfileBindingState:
    """Retain independent mismatch causes until each cause recovers."""

    def __init__(self) -> None:
        self._active: set[str] = set()

    def set(self, reason: str, *, active: bool) -> None:
        if not reason.startswith("profile:"):
            raise ValueError("profile mismatch reasons must use the profile namespace")
        if active:
            self._active.add(reason)
        else:
            self._active.discard(reason)

    @property
    def mismatched(self) -> bool:
        return bool(self._active)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    @property
    def reason_text(self) -> str:
        return ",".join(self.reasons)

    def blocking_reason(self, *, excluding: tuple[str, ...] = ()) -> str:
        excluded = set(excluding)
        return ",".join(
            reason for reason in self.reasons if reason not in excluded
        )


@dataclass(frozen=True)
class CameraProfile:
    width: int
    height: int
    rgb_encoding: str
    depth_encoding: str
    rate_hz: float
    validated_rate_range_hz: tuple[float, float]
    frame_id: str

    @property
    def image_profile(self) -> str:
        rate = int(self.rate_hz) if self.rate_hz.is_integer() else self.rate_hz
        return f"{self.width}x{self.height}@{rate}Hz"


@dataclass(frozen=True)
class ResourceBudget:
    processing_p95_ms: float
    depth_geometry_average_cpu_cores: float
    soak_duration_seconds: int
    maximum_input_queue_depth: int
    maximum_rgb_reorder_depth: int
    maximum_memory_growth_bytes: int
    measurement_window_samples: int


@dataclass(frozen=True)
class DetectionProfile:
    schema_version: int
    profile_id: str
    validation_phase: str
    maximum_speed_mps: float
    camera: CameraProfile
    observed_camera_height_m: float
    validated_height_range_m: tuple[float, float]
    observed_downward_pitch_degrees: float
    validated_downward_pitch_range_degrees: tuple[float, float]
    footprint_m: Mapping[str, float]
    rule_version: str
    model_version: str
    parameters: Mapping[str, object]
    resource_budget: ResourceBudget
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "DetectionProfile":
        source = Path(path)
        try:
            raw = source.read_bytes()
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load detection profile {source}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("detection profile root must be an object")
        try:
            validation = _mapping(document, "validation")
            camera = _mapping(document, "camera")
            installation = _mapping(document, "installation")
            implementation = _mapping(document, "implementation")
            budget = _mapping(document, "resource_budget")
            parameters = _mapping(document, "parameters")
            height_range = installation["validated_height_range_m"]
            if not isinstance(height_range, list) or len(height_range) != 2:
                raise ValueError(
                    "installation.validated_height_range_m must have two values"
                )
            rate_range = camera["validated_rate_range_hz"]
            if not isinstance(rate_range, list) or len(rate_range) != 2:
                raise ValueError(
                    "camera.validated_rate_range_hz must have two values"
                )
            pitch_range = installation[
                "validated_downward_pitch_range_degrees"
            ]
            if not isinstance(pitch_range, list) or len(pitch_range) != 2:
                raise ValueError(
                    "installation.validated_downward_pitch_range_degrees "
                    "must have two values"
                )
            profile = cls(
                schema_version=int(document["schema_version"]),
                profile_id=str(document["profile_id"]),
                validation_phase=str(validation["phase"]),
                maximum_speed_mps=float(validation["maximum_speed_mps"]),
                camera=CameraProfile(
                    width=int(camera["width"]),
                    height=int(camera["height"]),
                    rgb_encoding=str(camera["rgb_encoding"]),
                    depth_encoding=str(camera["depth_encoding"]),
                    rate_hz=float(camera["rate_hz"]),
                    validated_rate_range_hz=(
                        float(rate_range[0]),
                        float(rate_range[1]),
                    ),
                    frame_id=str(camera["frame_id"]),
                ),
                observed_camera_height_m=float(
                    installation["observed_camera_height_m"]
                ),
                validated_height_range_m=(
                    float(height_range[0]),
                    float(height_range[1]),
                ),
                observed_downward_pitch_degrees=float(
                    installation["observed_downward_pitch_degrees"]
                ),
                validated_downward_pitch_range_degrees=(
                    float(pitch_range[0]),
                    float(pitch_range[1]),
                ),
                footprint_m={
                    str(key): float(value)
                    for key, value in _mapping(
                        installation, "footprint_m"
                    ).items()
                },
                rule_version=str(implementation["rule_version"]),
                model_version=str(implementation["model_version"]),
                parameters=dict(parameters),
                resource_budget=ResourceBudget(
                    processing_p95_ms=float(budget["processing_p95_ms"]),
                    depth_geometry_average_cpu_cores=float(
                        budget["depth_geometry_average_cpu_cores"]
                    ),
                    soak_duration_seconds=int(budget["soak_duration_seconds"]),
                    maximum_input_queue_depth=int(
                        budget["maximum_input_queue_depth"]
                    ),
                    maximum_rgb_reorder_depth=int(
                        budget["maximum_rgb_reorder_depth"]
                    ),
                    maximum_memory_growth_bytes=int(
                        budget.get("maximum_memory_growth_bytes", 32 * 1024 * 1024)
                    ),
                    measurement_window_samples=int(
                        budget.get("measurement_window_samples", 256)
                    ),
                ),
                fingerprint=hashlib.sha256(
                    json.dumps(
                        document, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid detection profile {source}: {error}") from error
        profile._validate_document()
        return profile

    def _validate_document(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported detection profile schema {self.schema_version}"
            )
        if not self.profile_id or not self.rule_version:
            raise ValueError("profile_id and rule_version are required")
        if self.validation_phase != "home_feasibility":
            raise ValueError("the initial profile must remain home feasibility")
        lower, upper = self.validated_height_range_m
        minimum_rate, maximum_rate = self.camera.validated_rate_range_hz
        minimum_pitch, maximum_pitch = (
            self.validated_downward_pitch_range_degrees
        )
        numeric = (
            self.maximum_speed_mps,
            self.camera.rate_hz,
            self.observed_camera_height_m,
            lower,
            upper,
            minimum_rate,
            maximum_rate,
            minimum_pitch,
            maximum_pitch,
            self.resource_budget.processing_p95_ms,
            self.resource_budget.depth_geometry_average_cpu_cores,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("profile numeric limits must be finite and positive")
        if lower > self.observed_camera_height_m or upper < self.observed_camera_height_m:
            raise ValueError("observed camera height must be inside its validated range")
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if not minimum_rate <= self.camera.rate_hz <= maximum_rate:
            raise ValueError("camera rate must be inside its validated range")
        if not (
            minimum_pitch
            <= self.observed_downward_pitch_degrees
            <= maximum_pitch
        ):
            raise ValueError("camera pitch must be inside its validated range")
        if self.resource_budget.maximum_input_queue_depth != 1:
            raise ValueError("the initial profile requires latest-only input queues")
        if self.resource_budget.maximum_rgb_reorder_depth < 2:
            raise ValueError("RGB event-time reorder depth must be at least two")
        structural_bindings = {
            "expected_width": self.camera.width,
            "expected_height": self.camera.height,
            "camera_frame": self.camera.frame_id,
            "operating_speed_mps": self.maximum_speed_mps,
            "rgb_reorder_capacity": (
                self.resource_budget.maximum_rgb_reorder_depth
            ),
        }
        for name, expected in structural_bindings.items():
            if name not in self.parameters or not _same_value(
                self.parameters[name], expected
            ):
                raise ValueError(
                    f"profile parameter {name!r} must match profile metadata"
                )
        if self.resource_budget.soak_duration_seconds < 7200:
            raise ValueError("the profile soak budget must be at least two hours")
        if self.resource_budget.maximum_memory_growth_bytes <= 0:
            raise ValueError("memory-growth budget must be positive")
        if self.resource_budget.measurement_window_samples < 2:
            raise ValueError("measurement window must retain at least two samples")

    def validate_image(
        self, *, width: int, height: int, encoding: str, is_depth: bool
    ) -> None:
        expected_encoding = (
            self.camera.depth_encoding if is_depth else self.camera.rgb_encoding
        )
        if (width, height, encoding) != (
            self.camera.width,
            self.camera.height,
            expected_encoding,
        ):
            raise ProfileMismatch(
                "image profile mismatch: "
                f"delivered {width}x{height}/{encoding}, profile "
                f"requires {self.camera.width}x{self.camera.height}/"
                f"{expected_encoding}"
            )

    def validate_observed_camera_height(self, height_m: float) -> None:
        lower, upper = self.validated_height_range_m
        if not math.isfinite(height_m) or not lower <= height_m <= upper:
            raise ProfileMismatch(
                "camera installation mismatch: observed height "
                f"{height_m:.3f} m is outside [{lower:.3f}, {upper:.3f}] m"
            )

    def validate_rate(self, rate_hz: float) -> None:
        lower, upper = self.camera.validated_rate_range_hz
        if not math.isfinite(rate_hz) or not lower <= rate_hz <= upper:
            raise ProfileMismatch(
                "image rate mismatch: observed "
                f"{rate_hz:.3f} Hz is outside [{lower:.3f}, {upper:.3f}] Hz"
            )

    def validate_observed_downward_pitch(self, pitch_degrees: float) -> None:
        lower, upper = self.validated_downward_pitch_range_degrees
        if not math.isfinite(pitch_degrees) or not lower <= pitch_degrees <= upper:
            raise ProfileMismatch(
                "camera installation mismatch: observed downward pitch "
                f"{pitch_degrees:.3f} degrees is outside "
                f"[{lower:.3f}, {upper:.3f}] degrees"
            )

    def validate_speed(self, speed_mps: float) -> None:
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ProfileMismatch("speed must be finite and non-negative")
        if speed_mps > self.maximum_speed_mps + 1e-9:
            raise ProfileMismatch(
                f"speed {speed_mps:.3f} m/s exceeds profile maximum "
                f"{self.maximum_speed_mps:.3f} m/s"
            )

    def validate_parameters(
        self,
        runtime: Mapping[str, object],
        *,
        names: tuple[str, ...] | None = None,
    ) -> None:
        selected = self.parameters if names is None else {
            name: self.parameters[name] for name in names
        }
        for name, expected in selected.items():
            if name not in runtime:
                raise ProfileMismatch(f"profile parameter {name!r} is missing")
            actual = runtime[name]
            if not _same_value(actual, expected):
                raise ProfileMismatch(
                    f"profile parameter {name!r} is {actual!r}; "
                    f"validated value is {expected!r}"
                )


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
    return actual == expected


def default_detection_profile_path() -> Path:
    """Locate the installed profile, with a source-tree fallback for tests."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = (
            Path(get_package_share_directory("low_profile_hazard_perception"))
            / "config"
            / "detection_profile_dcw2_home_640x360_v1.json"
        )
        if installed.exists():
            return installed
    except (ImportError, LookupError):
        pass
    source = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "detection_profile_dcw2_home_640x360_v1.json"
    )
    if not source.exists():
        raise FileNotFoundError("default detection profile is not installed")
    return source
