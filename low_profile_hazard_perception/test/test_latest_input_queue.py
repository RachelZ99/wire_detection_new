import unittest

from low_profile_hazard_perception.latest_input_queue import LatestInputQueue


class LatestInputQueueTests(unittest.TestCase):
    def test_new_input_replaces_old_work_and_counts_the_drop(self) -> None:
        queue: LatestInputQueue[str] = LatestInputQueue()

        self.assertFalse(queue.offer("old", sensor_stamp_ns=1_000_000_000))
        self.assertTrue(queue.offer("latest", sensor_stamp_ns=1_100_000_000))

        self.assertEqual(queue.take(), "latest")
        self.assertIsNone(queue.take())
        self.assertEqual(queue.received_count, 2)
        self.assertEqual(queue.processed_count, 1)
        self.assertEqual(queue.drop_count, 1)
        self.assertEqual(queue.approximate_received_rate_hz, 10.0)


if __name__ == "__main__":
    unittest.main()
