import unittest

from low_profile_hazard_perception.latest_input_queue import LatestInputQueue


class LatestInputQueueTests(unittest.TestCase):
    def test_delivered_rate_window_tolerates_arrival_reordering(self) -> None:
        queue: LatestInputQueue[str] = LatestInputQueue()
        for stamp_ns in (
            1_000_000_000,
            1_300_000_000,
            1_100_000_000,
            1_200_000_000,
        ):
            queue.offer("frame", sensor_stamp_ns=stamp_ns)

        self.assertEqual(queue.stamped_window_count, 4)
        self.assertEqual(queue.approximate_received_rate_hz, 10.0)

    def test_delivered_rate_window_resets_on_a_new_replay_epoch(self) -> None:
        queue: LatestInputQueue[str] = LatestInputQueue()
        for stamp_ns in (
            10_000_000_000,
            10_100_000_000,
            10_200_000_000,
            1_000_000_000,
            1_100_000_000,
            1_200_000_000,
        ):
            queue.offer("frame", sensor_stamp_ns=stamp_ns)

        self.assertEqual(queue.stamped_window_count, 3)
        self.assertEqual(queue.approximate_received_rate_hz, 10.0)

    def test_new_input_replaces_old_work_and_counts_the_drop(self) -> None:
        queue: LatestInputQueue[str] = LatestInputQueue()

        self.assertFalse(queue.offer("old", sensor_stamp_ns=1_000_000_000))
        self.assertTrue(queue.offer("latest", sensor_stamp_ns=1_100_000_000))
        self.assertEqual(queue.pending_count, 1)

        self.assertEqual(queue.take(), "latest")
        self.assertEqual(queue.pending_count, 0)
        self.assertIsNone(queue.take())
        self.assertEqual(queue.received_count, 2)
        self.assertEqual(queue.processed_count, 1)
        self.assertEqual(queue.drop_count, 1)
        self.assertEqual(queue.approximate_received_rate_hz, 10.0)


if __name__ == "__main__":
    unittest.main()
