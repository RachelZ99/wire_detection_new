import math
import unittest

from low_profile_hazard_perception.temporal import (
    EvidenceSource,
    HazardObservation,
    HazardTracker,
    HazardTrackerConfig,
    OdomPoseCache,
    Pose3,
)


def _yaw(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


class TemporalObservationAlignmentTests(unittest.TestCase):
    def test_odom_pose_is_interpolated_at_the_observation_timestamp(
        self,
    ) -> None:
        cache = OdomPoseCache()
        cache.add(
            1_000_000_000,
            Pose3(translation=(0.0, 0.0, 0.0), rotation=_yaw(0.0)),
        )
        cache.add(
            1_100_000_000,
            Pose3(translation=(0.2, 0.0, 0.0), rotation=_yaw(20.0)),
        )

        pose = cache.interpolate(1_050_000_000)

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.translation[0], 0.1)
        transformed = pose.transform_point((1.0, 0.0, 0.0))
        self.assertAlmostEqual(
            transformed[0], 0.1 + math.cos(math.radians(10.0)), places=6
        )
        self.assertAlmostEqual(
            transformed[1], math.sin(math.radians(10.0)), places=6
        )

    def test_motion_aligned_observations_confirm_without_an_odom_trail(
        self,
    ) -> None:
        tracker = HazardTracker(
            HazardTrackerConfig(
                association_radius_m=0.08,
                confirmation_window_ns=350_000_000,
            )
        )
        first_pose = Pose3.identity()
        second_pose = Pose3(translation=(0.10, 0.0, 0.0), rotation=_yaw(0.0))
        world_point = (1.20, 0.05, 0.03)
        first_in_base = first_pose.inverse().transform_point(world_point)
        second_in_base = second_pose.inverse().transform_point(world_point)

        first = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=(first_pose.transform_point(first_in_base),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.9,
            )
        )
        second = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_100_000_000,
                points_odom=(second_pose.transform_point(second_in_base),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.9,
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(len(second), 1)
        confirmed = second[0]
        self.assertEqual(confirmed.observation_count, 2)
        self.assertEqual(confirmed.sensor_stamp_ns, 1_100_000_000)
        self.assertAlmostEqual(confirmed.centroid[0], world_point[0], places=6)
        self.assertLess(confirmed.spatial_spread_m, 1e-6)

    def test_one_isolated_observation_never_enters_operational_output(
        self,
    ) -> None:
        tracker = HazardTracker()

        confirmed = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=((0.8, 0.2, 0.04),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.95,
            )
        )

        self.assertEqual(confirmed, ())

    def test_confirmed_hazard_outlives_candidate_and_is_not_duplicated(
        self,
    ) -> None:
        tracker = HazardTracker(
            HazardTrackerConfig(
                association_radius_m=0.08,
                confirmation_window_ns=350_000_000,
                candidate_retention_ns=500_000_000,
                confirmed_retention_ns=2_000_000_000,
            )
        )
        first = HazardObservation(
            sensor_stamp_ns=1_000_000_000,
            points_odom=((0.8, 0.2, 0.04),),
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.9,
        )
        second = HazardObservation(
            sensor_stamp_ns=1_100_000_000,
            points_odom=((0.81, 0.2, 0.04),),
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.95,
        )

        self.assertEqual(tracker.observe(first), ())
        self.assertEqual(len(tracker.observe(second)), 1)

        # The candidate lifetime has elapsed, but the confirmed hazard remains
        # available throughout the observation blind zone retention interval.
        retained = tracker.retained_at(1_700_000_000)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].observation_count, 2)

        recovered = HazardObservation(
            sensor_stamp_ns=1_800_000_000,
            points_odom=((0.805, 0.2, 0.04),),
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.92,
        )
        self.assertEqual(len(tracker.observe(recovered)), 1)
        retained = tracker.retained_at(1_800_000_000)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].observation_count, 3)

        self.assertEqual(
            len(
                tracker.retained_at(
                    3_800_000_001, allow_confirmed_expiry=False
                )
            ),
            1,
        )
        self.assertEqual(tracker.retained_at(3_800_000_001), ())

    def test_unconfirmed_candidate_uses_its_own_short_retention(self) -> None:
        tracker = HazardTracker(
            HazardTrackerConfig(
                candidate_retention_ns=500_000_000,
                confirmed_retention_ns=2_000_000_000,
            )
        )
        tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=((0.8, 0.2, 0.04),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.9,
            )
        )

        self.assertEqual(tracker.candidate_count_at(1_500_000_000), 1)
        self.assertEqual(tracker.candidate_count_at(1_500_000_001), 0)
        self.assertEqual(tracker.retained_at(1_500_000_001), ())

    def test_stale_or_discontinuous_odom_cannot_support_alignment(
        self,
    ) -> None:
        stale = OdomPoseCache()
        stale.add(1_000_000_000, Pose3.identity())
        stale.add(
            1_300_000_000,
            Pose3(translation=(0.03, 0.0, 0.0), rotation=_yaw(0.0)),
        )
        self.assertIsNone(stale.interpolate(1_150_000_000))
        self.assertIsNone(stale.interpolate(1_000_000_000))
        self.assertFalse(
            stale.continuous_between(1_000_000_000, 1_300_000_000)
        )

        jumped = OdomPoseCache()
        jumped.add(1_000_000_000, Pose3.identity())
        jumped.add(
            1_080_000_000,
            Pose3(translation=(0.50, 0.0, 0.0), rotation=_yaw(0.0)),
        )
        self.assertIsNone(jumped.interpolate(1_040_000_000))

    def test_odom_alignment_reports_missing_stale_and_discontinuous_support(
        self,
    ) -> None:
        cache = OdomPoseCache()
        self.assertEqual(
            cache.alignment_at(1_000_000_000).reason, "odom:missing"
        )
        cache.add(1_000_000_000, Pose3.identity())
        cache.add(1_080_000_000, Pose3.identity())

        self.assertEqual(
            cache.alignment_at(1_200_000_000).reason, "odom:stale"
        )

        discontinuous = OdomPoseCache()
        discontinuous.add(1_000_000_000, Pose3.identity())
        discontinuous.add(1_300_000_000, Pose3.identity())
        self.assertEqual(
            discontinuous.alignment_at(1_150_000_000).reason,
            "odom:discontinuous",
        )

    def test_disordered_odom_is_rejected_with_a_reason(self) -> None:
        cache = OdomPoseCache()
        self.assertEqual(cache.add(1_000_000_000, Pose3.identity()), "")
        self.assertEqual(
            cache.add(
                1_100_000_000,
                Pose3((0.1, 0.0, 0.0), Pose3.identity().rotation),
            ),
            "",
        )

        reason = cache.add(
            1_050_000_000,
            Pose3((9.0, 0.0, 0.0), Pose3.identity().rotation),
        )

        self.assertEqual(reason, "odom:disordered")
        pose = cache.interpolate(1_050_000_000)
        assert pose is not None
        self.assertAlmostEqual(pose.translation[0], 0.05)


if __name__ == "__main__":
    unittest.main()
