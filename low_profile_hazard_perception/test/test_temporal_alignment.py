import math
import unittest

from low_profile_hazard_perception.temporal import (
    CandidateDecisionReason,
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
        self.assertAlmostEqual(transformed[1], math.sin(math.radians(10.0)), places=6)

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
        self.assertEqual(confirmed.confirmation_latency_ns, 100_000_000)
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

    def test_two_candidates_from_one_sensor_stamp_are_not_independent(self) -> None:
        tracker = HazardTracker()

        first = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=((0.80, 0.20, 0.04),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.90,
            )
        )
        same_frame = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=((0.81, 0.20, 0.04),),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.95,
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(same_frame, ())
        self.assertEqual(tracker.retained_at(1_000_000_000), ())

    def test_repeated_invalid_depth_cannot_confirm_without_cable_shape(
        self,
    ) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.0),)

        first = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=point,
                evidence=EvidenceSource.INVALID_DEPTH,
                confidence=0.7,
            )
        )
        second = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_100_000_000,
                points_odom=point,
                evidence=EvidenceSource.INVALID_DEPTH,
                confidence=0.8,
            )
        )

        self.assertEqual(first, ())
        self.assertEqual(second, ())
        self.assertEqual(tracker.retained_at(1_100_000_000), ())

    def test_weak_height_strengthens_a_matching_cable_but_does_not_count(
        self,
    ) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.01),)

        weak = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=point,
                evidence=EvidenceSource.WEAK_HEIGHT,
                confidence=0.6,
            )
        )
        first_rgb = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_100_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.8,
            )
        )
        second_rgb = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_200_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.82,
            )
        )

        self.assertEqual(weak, ())
        self.assertEqual(first_rgb, ())
        self.assertEqual(len(second_rgb), 1)
        self.assertEqual(second_rgb[0].observation_count, 2)
        self.assertEqual(
            second_rgb[0].evidence,
            (EvidenceSource.RGB_CABLE, EvidenceSource.WEAK_HEIGHT),
        )

    def test_low_confidence_rgb_shape_cannot_confirm_by_repetition(self) -> None:
        tracker = HazardTracker(
            HazardTrackerConfig(minimum_rgb_confirmation_confidence=0.75)
        )
        point = ((0.8, 0.2, 0.0),)

        for stamp_ns in (1_000_000_000, 1_100_000_000):
            confirmed = tracker.observe(
                HazardObservation(
                    sensor_stamp_ns=stamp_ns,
                    points_odom=point,
                    evidence=EvidenceSource.RGB_CABLE,
                    confidence=0.70,
                )
            )
            self.assertEqual(confirmed, ())

        self.assertEqual(tracker.retained_at(1_100_000_000), ())

    def test_mixed_evidence_converges_under_different_arrival_orders(
        self,
    ) -> None:
        point = ((0.8, 0.2, 0.01),)
        observations = (
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.80,
            ),
            HazardObservation(
                sensor_stamp_ns=1_050_000_000,
                points_odom=point,
                evidence=EvidenceSource.WEAK_HEIGHT,
                confidence=0.60,
            ),
            HazardObservation(
                sensor_stamp_ns=1_080_000_000,
                points_odom=point,
                evidence=EvidenceSource.INVALID_DEPTH,
                confidence=0.55,
            ),
            HazardObservation(
                sensor_stamp_ns=1_200_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.85,
            ),
        )

        outcomes = []
        for order in (observations, tuple(reversed(observations))):
            tracker = HazardTracker()
            for observation in order:
                tracker.observe(observation)
            outcomes.append(tracker.retained_at(1_200_000_000))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(len(outcomes[0]), 1)
        self.assertEqual(outcomes[0][0].sensor_stamp_ns, 1_200_000_000)
        self.assertEqual(outcomes[0][0].observation_count, 2)
        self.assertEqual(
            outcomes[0][0].evidence,
            (
                EvidenceSource.INVALID_DEPTH,
                EvidenceSource.RGB_CABLE,
                EvidenceSource.WEAK_HEIGHT,
            ),
        )

    def test_invalid_depth_can_strengthen_matching_strong_geometry(self) -> None:
        point = ((0.8, 0.2, 0.02),)
        invalid = HazardObservation(
            sensor_stamp_ns=1_050_000_000,
            points_odom=point,
            evidence=EvidenceSource.INVALID_DEPTH,
            confidence=0.55,
        )
        first_geometry = HazardObservation(
            sensor_stamp_ns=1_000_000_000,
            points_odom=point,
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.90,
        )
        second_geometry = HazardObservation(
            sensor_stamp_ns=1_200_000_000,
            points_odom=point,
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.92,
        )

        outcomes = []
        for observations in (
            (invalid, first_geometry, second_geometry),
            (first_geometry, invalid, second_geometry),
        ):
            tracker = HazardTracker()
            decisions = [
                tracker.observe_with_decision(observation)
                for observation in observations
            ]
            outcomes.append(tracker.retained_at(1_200_000_000))
            invalid_decision = decisions[observations.index(invalid)]
            self.assertEqual(
                invalid_decision.decision_reason,
                CandidateDecisionReason.SUPPORT_ONLY,
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(len(outcomes[0]), 1)
        self.assertEqual(outcomes[0][0].observation_count, 2)
        self.assertEqual(
            outcomes[0][0].evidence,
            (
                EvidenceSource.INVALID_DEPTH,
                EvidenceSource.STRONG_GEOMETRY,
            ),
        )

    def test_tracker_exposes_machine_readable_mixed_evidence_decisions(
        self,
    ) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.01),)

        support = tracker.observe_with_decision(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=point,
                evidence=EvidenceSource.WEAK_HEIGHT,
                confidence=0.60,
            )
        )
        pending = tracker.observe_with_decision(
            HazardObservation(
                sensor_stamp_ns=1_100_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.80,
            )
        )
        confirmed = tracker.observe_with_decision(
            HazardObservation(
                sensor_stamp_ns=1_200_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.85,
            )
        )

        self.assertEqual(support.decision_reason, CandidateDecisionReason.SUPPORT_ONLY)
        self.assertEqual(
            pending.decision_reason,
            CandidateDecisionReason.WAITING_FOR_CONFIRMATION,
        )
        self.assertEqual(
            confirmed.decision_reason,
            CandidateDecisionReason.CONFIRMED_MIXED_EVIDENCE,
        )
        self.assertEqual(
            confirmed.evidence,
            (EvidenceSource.RGB_CABLE, EvidenceSource.WEAK_HEIGHT),
        )
        self.assertGreater(confirmed.confidence, 0.85)
        self.assertEqual(len(confirmed.confirmed), 1)

    def test_late_support_does_not_replace_the_operational_source_stamp(
        self,
    ) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.01),)
        for stamp_ns in (1_000_000_000, 1_200_000_000):
            tracker.observe(
                HazardObservation(
                    sensor_stamp_ns=stamp_ns,
                    points_odom=point,
                    evidence=EvidenceSource.RGB_CABLE,
                    confidence=0.85,
                )
            )

        support = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_300_000_000,
                points_odom=point,
                evidence=EvidenceSource.INVALID_DEPTH,
                confidence=0.55,
            )
        )
        retained = tracker.retained_at(1_300_000_000)

        self.assertEqual(support, ())
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].sensor_stamp_ns, 1_200_000_000)
        self.assertIn(EvidenceSource.INVALID_DEPTH, retained[0].evidence)

    def test_support_during_recovery_strengthens_the_retained_track(self) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.01),)
        for stamp_ns in (1_000_000_000, 1_100_000_000):
            tracker.observe(
                HazardObservation(
                    sensor_stamp_ns=stamp_ns,
                    points_odom=point,
                    evidence=EvidenceSource.RGB_CABLE,
                    confidence=0.85,
                )
            )

        support = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_200_000_000,
                points_odom=point,
                evidence=EvidenceSource.WEAK_HEIGHT,
                confidence=0.60,
            ),
            allow_confirmed_expiry=False,
            require_reconfirmation_for_confirmed=True,
        )
        retained = tracker.retained_at(1_200_000_000, allow_confirmed_expiry=False)

        self.assertEqual(support, ())
        self.assertEqual(len(retained), 1)
        self.assertIn(EvidenceSource.WEAK_HEIGHT, retained[0].evidence)
        self.assertEqual(tracker.candidate_count_at(1_200_000_000), 0)

    def test_recovery_reconfirmation_is_arrival_order_independent(self) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.01),)
        for stamp_ns in (1_000_000_000, 1_100_000_000):
            tracker.observe(
                HazardObservation(
                    sensor_stamp_ns=stamp_ns,
                    points_odom=point,
                    evidence=EvidenceSource.STRONG_GEOMETRY,
                    confidence=0.9,
                )
            )

        outputs = []
        for stamp_ns in (1_400_000_000, 1_300_000_000):
            outputs.extend(
                tracker.observe(
                    HazardObservation(
                        sensor_stamp_ns=stamp_ns,
                        points_odom=point,
                        evidence=EvidenceSource.STRONG_GEOMETRY,
                        confidence=0.92,
                    ),
                    allow_confirmed_expiry=False,
                    require_reconfirmation_for_confirmed=True,
                )
            )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].sensor_stamp_ns, 1_400_000_000)
        self.assertEqual(len(tracker.retained_at(1_400_000_000)), 1)

    def test_independent_geometry_and_rgb_observations_cross_confirm(self) -> None:
        tracker = HazardTracker()
        point = ((0.8, 0.2, 0.03),)

        geometry = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_000_000_000,
                points_odom=point,
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.9,
            )
        )
        first_rgb = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_100_000_000,
                points_odom=point,
                evidence=EvidenceSource.RGB_CABLE,
                confidence=0.9,
            )
        )

        self.assertEqual(geometry, ())
        self.assertEqual(len(first_rgb), 1)
        self.assertEqual(first_rgb[0].observation_count, 2)
        self.assertEqual(
            first_rgb[0].evidence,
            (EvidenceSource.RGB_CABLE, EvidenceSource.STRONG_GEOMETRY),
        )

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
            len(tracker.retained_at(3_800_000_001, allow_confirmed_expiry=False)),
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

    def test_recovery_requires_two_observations_before_refreshing_shape(
        self,
    ) -> None:
        tracker = HazardTracker()
        for stamp_ns, x in (
            (1_000_000_000, 0.80),
            (1_100_000_000, 0.81),
        ):
            tracker.observe(
                HazardObservation(
                    sensor_stamp_ns=stamp_ns,
                    points_odom=((x, 0.2, 0.04),),
                    evidence=EvidenceSource.STRONG_GEOMETRY,
                    confidence=0.9,
                )
            )
        original = tracker.retained_at(1_100_000_000)[0]
        first_refresh = HazardObservation(
            sensor_stamp_ns=1_800_000_000,
            points_odom=((0.86, 0.2, 0.04), (0.87, 0.2, 0.04)),
            evidence=EvidenceSource.STRONG_GEOMETRY,
            confidence=0.95,
        )

        first_result = tracker.observe(
            first_refresh,
            allow_confirmed_expiry=False,
            require_reconfirmation_for_confirmed=True,
        )

        self.assertEqual(first_result, ())
        after_first = tracker.retained_at(1_800_000_000, allow_confirmed_expiry=False)[
            0
        ]
        self.assertEqual(after_first.points_odom, original.points_odom)
        second_result = tracker.observe(
            HazardObservation(
                sensor_stamp_ns=1_900_000_000,
                points_odom=((0.855, 0.2, 0.04), (0.865, 0.2, 0.04)),
                evidence=EvidenceSource.STRONG_GEOMETRY,
                confidence=0.96,
            ),
            allow_confirmed_expiry=False,
            require_reconfirmation_for_confirmed=True,
        )
        self.assertEqual(len(second_result), 1)
        self.assertNotEqual(second_result[0].points_odom, original.points_odom)

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
        self.assertFalse(stale.continuous_between(1_000_000_000, 1_300_000_000))

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
        self.assertEqual(cache.alignment_at(1_000_000_000).reason, "odom:missing")
        cache.add(1_000_000_000, Pose3.identity())
        cache.add(1_080_000_000, Pose3.identity())

        self.assertEqual(cache.alignment_at(1_200_000_000).reason, "odom:stale")

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
