"""Pure acceptance checks for training-free RGB cable replay output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def audit_rgb_cable_replay(
    *,
    stable_values: Mapping[str, str],
    clouds: Sequence[Mapping[str, Any]],
    maximum_alignment_spread_m: float,
    minimum_physical_span_m: float,
    expected_center_odom: tuple[float, float] | None = None,
    expected_center_radius_m: float = 0.15,
) -> dict[str, int | float]:
    """Require formal cable evidence to reach a supported odom cloud."""
    if stable_values.get("cable.provider") != "training_free_thin_line":
        raise ValueError("replay did not use the training-free cable provider")
    if stable_values.get("cable.rgb_depth_synchronizer") != "disabled":
        raise ValueError("RGB replay used a forbidden RGB-depth synchronizer")
    if stable_values.get("cable.diagnostic_pink_operational") != "false":
        raise ValueError("diagnostic pink comparison entered operational output")
    if stable_values.get("cable.confirmation_observations") != "2":
        raise ValueError("cable replay did not use two-observation confirmation")
    try:
        confirmed_count = int(
            stable_values["cable.confirmed_observation_count"]
        )
        processed_rgb_count = int(stable_values["cable.processed_rgb_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("replay is missing cable health counters") from error
    if confirmed_count < 1:
        raise ValueError("replay produced no two-observation cable confirmation")
    if processed_rgb_count < 2:
        raise ValueError("replay processed fewer than two RGB observations")

    hazard_clouds = [cloud for cloud in clouds if not cloud.get("clearing")]
    supported_clouds = [
        cloud
        for cloud in hazard_clouds
        if cloud.get("frame_id") == "odom"
        and int(cloud.get("stamp_ns") or 0) > 0
        and cloud.get("source_stamp_min_ns") == cloud.get("stamp_ns")
        and cloud.get("confirmation_spread_m") is not None
        and float(cloud["confirmation_spread_m"])
        <= maximum_alignment_spread_m
        and int(cloud.get("rgb_cable_point_count") or 0) > 0
        and float(cloud.get("rgb_cable_span_m") or 0.0)
        >= minimum_physical_span_m
    ]
    if expected_center_odom is not None:
        expected_x, expected_y = expected_center_odom
        supported_clouds = [
            cloud
            for cloud in supported_clouds
            if cloud.get("rgb_cable_centroid_x_m") is not None
            and cloud.get("rgb_cable_centroid_y_m") is not None
            and (
                (float(cloud["rgb_cable_centroid_x_m"]) - expected_x) ** 2
                + (float(cloud["rgb_cable_centroid_y_m"]) - expected_y) ** 2
            )
            ** 0.5
            <= expected_center_radius_m
        ]
    if not supported_clouds:
        raise ValueError(
            "replay produced no physically supported, aligned odom cable cloud"
        )
    maximum_spread = max(
        float(cloud["confirmation_spread_m"]) for cloud in supported_clouds
    )
    return {
        "confirmed_cable_event_count": confirmed_count,
        "processed_rgb_count": processed_rgb_count,
        "supported_odom_cloud_count": len(supported_clouds),
        "maximum_alignment_spread_m": maximum_spread,
    }
