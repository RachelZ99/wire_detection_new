"""Pure replay-result aggregation shared by ROS replay and tests."""

from __future__ import annotations

from typing import Any


TIMING_FIELD_SUFFIXES = (
    ".sensor_stamp_age_ms",
    ".receive_age_ms",
    ".processing_latency_ms",
)


def is_timing_field(key: str) -> bool:
    return (key.startswith("stage.") and key.endswith("_ms")) or any(
        key == suffix.removeprefix(".") or key.endswith(suffix)
        for suffix in TIMING_FIELD_SUFFIXES
    )


def is_volatile_field(key: str) -> bool:
    """Fields measured from the host must not define replay determinism."""
    return (
        is_timing_field(key)
        or key.startswith("resource.")
        or key.startswith("stage.")
        or key == "budget.runtime_reasons"
    )


class ReplayResultAccumulator:
    """Retain health transitions and timing while comparing stable fields."""

    def __init__(self, *, diagnostic_name: str) -> None:
        self._diagnostic_name = diagnostic_name
        self._latest_values: dict[str, str] | None = None
        self._latest_state = ""
        self._transitions: list[dict[str, str]] = []
        self._timing_samples: dict[str, list[float]] = {}
        self._invalid_observed = False

    def record(self, *, state: str, values: dict[str, str]) -> None:
        self._latest_state = state
        self._latest_values = dict(values)
        self._invalid_observed = self._invalid_observed or state == "INVALID"
        transition = {"state": state, "reasons": values.get("reasons", "")}
        if not self._transitions or self._transitions[-1] != transition:
            self._transitions.append(transition)
        for key, value in values.items():
            if not is_timing_field(key) or value == "unknown":
                continue
            try:
                sample = float(value)
            except ValueError:
                continue
            self._timing_samples.setdefault(key, []).append(sample)

    def report(self) -> dict[str, Any]:
        if self._latest_values is None:
            raise RuntimeError("replay produced no perception-health result")
        stable_values = {
            key: value
            for key, value in sorted(self._latest_values.items())
            if not is_volatile_field(key)
        }
        latest_volatile_values = {
            key: value
            for key, value in sorted(self._latest_values.items())
            if is_volatile_field(key)
        }
        timing_ranges = {
            key: {
                "minimum": min(samples),
                "maximum": max(samples),
                "last": samples[-1],
            }
            for key, samples in sorted(self._timing_samples.items())
        }
        return {
            "canonical": {
                "diagnostic_name": self._diagnostic_name,
                "state": self._latest_state,
                "stable_values": stable_values,
                "timing_fields_present": sorted(self._timing_samples),
            },
            "transitions": list(self._transitions),
            "timing_ranges_ms": timing_ranges,
            "latest_volatile_values": latest_volatile_values,
            "invalid_observed": self._invalid_observed,
        }
