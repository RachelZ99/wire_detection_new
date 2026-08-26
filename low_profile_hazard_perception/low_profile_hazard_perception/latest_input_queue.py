"""A bounded latest-only work queue with explicit drop accounting."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, TypeVar


Item = TypeVar("Item")
_MAXIMUM_STAMP_REORDER_NS = 500_000_000


class LatestInputQueue(Generic[Item]):
    """Keep one latest item and count every overwritten item."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._item: Item | None = None
        self._received_count = 0
        self._processed_count = 0
        self._drop_count = 0
        self._sensor_stamp_window: deque[int] = deque(maxlen=32)

    def offer(self, item: Item, *, sensor_stamp_ns: int | None = None) -> bool:
        """Offer work, returning true when pending work was dropped."""
        with self._lock:
            dropped = self._item is not None
            self._received_count += 1
            self._drop_count += int(dropped)
            self._item = item
            if sensor_stamp_ns is not None:
                if (
                    self._sensor_stamp_window
                    and sensor_stamp_ns
                    < max(self._sensor_stamp_window) - _MAXIMUM_STAMP_REORDER_NS
                ):
                    self._sensor_stamp_window.clear()
                self._sensor_stamp_window.append(sensor_stamp_ns)
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
    def pending_count(self) -> int:
        with self._lock:
            return int(self._item is not None)

    @property
    def approximate_received_rate_hz(self) -> float | None:
        with self._lock:
            stamps = sorted(set(self._sensor_stamp_window))
            if len(stamps) < 2 or stamps[-1] <= stamps[0]:
                return None
            return (
                (len(stamps) - 1) * 1_000_000_000 / (stamps[-1] - stamps[0])
            )

    @property
    def stamped_window_count(self) -> int:
        with self._lock:
            return len(set(self._sensor_stamp_window))
