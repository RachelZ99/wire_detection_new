"""Pure acceptance checks for geometric black-box replay output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


class GeometricReplayCloud(TypedDict):
    """Cloud timing fields needed by the observation-blind-zone audit."""

    clearing: bool
    source_stamp_max_ns: int | None
    stamp_ns: int


def observation_blind_zone_retention_audit(
    clouds: Sequence[GeometricReplayCloud],
    *,
    latest_processed_depth_stamp_ns: int,
    minimum_retention_ns: int,
) -> dict[str, int]:
    """Prove retention while usable depth continues through the blind zone."""
    observation_blind_zone_entry_ns: int | None = None
    completed_interval: tuple[int, int] | None = None
    for cloud in clouds:
        if cloud["clearing"]:
            if observation_blind_zone_entry_ns is not None:
                completed_interval = (
                    observation_blind_zone_entry_ns,
                    cloud["stamp_ns"],
                )
            continue
        source_stamp_ns = cloud["source_stamp_max_ns"]
        if source_stamp_ns is not None:
            observation_blind_zone_entry_ns = source_stamp_ns
            completed_interval = None

    if completed_interval is None:
        raise ValueError(
            "replay did not clear a confirmed hazard after observation "
            "blind-zone retention"
        )
    blind_zone_entry_ns, clearing_ns = completed_interval
    if latest_processed_depth_stamp_ns <= blind_zone_entry_ns:
        raise ValueError(
            "replay did not show continued depth processing after the "
            "hazard entered the observation blind zone"
        )
    retention_ns = clearing_ns - blind_zone_entry_ns
    if retention_ns < minimum_retention_ns:
        raise ValueError(
            "confirmed hazard retention was shorter than the conservative "
            f"minimum: {retention_ns} ns"
        )
    return {
        "observation_blind_zone_entry_ns": blind_zone_entry_ns,
        "clearing_ns": clearing_ns,
        "retention_ns": retention_ns,
        "depth_after_observation_blind_zone_entry_ns": (
            latest_processed_depth_stamp_ns - blind_zone_entry_ns
        ),
    }
