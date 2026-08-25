"""A bounded latest-only work queue with explicit drop accounting."""

from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar


Item = TypeVar("Item")


class LatestInputQueue(Generic[Item]):
    """Keep one latest item and count every overwritten item."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._item: Item | None = None
        self._received_count = 0
        self._processed_count = 0
        self._drop_count = 0
        self._first_sensor_stamp_ns: int | None = None
        self._last_sensor_stamp_ns: int | None = None
        self._stamped_count = 0

    def offer(self, item: Item, *, sensor_stamp_ns: int | None = None) -> bool:
        """Offer work, returning true when pending work was dropped."""
        with self._lock:
            dropped = self._item is not None
            self._received_count += 1
            self._drop_count += int(dropped)
            self._item = item
            if sensor_stamp_ns is not None:
                if self._first_sensor_stamp_ns is None:
                    self._first_sensor_stamp_ns = sensor_stamp_ns
                self._last_sensor_stamp_ns = sensor_stamp_ns
                self._stamped_count += 1
            return dropped

    def take(self) -> Item | None:
        """Take the latest work item, if one is pending."""
        with self._lock:
            item = self._item
            self._item = None
            self._processed_count += int(item is not None)
            return item

    @property
    def received_count(self) -> int:
        with self._lock:
            return self._received_count

    @property
    def processed_count(self) -> int:
        with self._lock:
            return self._processed_count

    @property
    def drop_count(self) -> int:
        with self._lock:
            return self._drop_count

    @property
    def approximate_received_rate_hz(self) -> float | None:
        with self._lock:
            if (
                self._stamped_count < 2
                or self._first_sensor_stamp_ns is None
                or self._last_sensor_stamp_ns is None
                or self._last_sensor_stamp_ns <= self._first_sensor_stamp_ns
            ):
                return None
            return (
                (self._stamped_count - 1)
                * 1_000_000_000
                / (self._last_sensor_stamp_ns - self._first_sensor_stamp_ns)
            )
