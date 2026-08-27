import json
import unittest
from pathlib import Path

from low_profile_hazard_perception.response_trial import (
    ResponseTrialRecorder,
    evaluate_response_trial,
)


NS = 1_000_000_000


def _trial_events(speed: float = 0.3) -> list[dict[str, object]]:
    return [
        {
            "event": "trial_started",
            "stamp_ns": 1 * NS,
            "trial_id": "straight-cable-001",
            "profile_id": "dcw2-home-640x360-v1",
            "planned_speed_mps": speed,
            "motion": "straight",
            "approach_angle_degrees": 90.0,
            "hazard_kind": "cable",
        },
        {
            "event": "confirmed_hazard",
            "stamp_ns": 2_200_000_000,
            "observation_stamp_ns": 2 * NS,
            "confirmed_stamp_ns": 2_100_000_000,
            "hazard_track_id": 7,
            "detection_distance_m": 0.8,
        },
        {
            "event": "unified_response_received",
            "stamp_ns": 2_250_000_000,
            "hazard_track_id": 7,
        },
        {
            "event": "response_started",
            "stamp_ns": 2_300_000_000,
            "hazard_track_id": 7,
            "response": "stop",
        },
        {
            "event": "command_sample",
            "stamp_ns": 2_250_000_000,
            "stream": "command",
            "linear_mps": speed,
            "angular_rps": 0.0,
        },
        {
            "event": "command_sample",
            "stamp_ns": 2_300_000_000,
            "stream": "smoothed_command",
            "linear_mps": speed,
            "angular_rps": 0.0,
        },
        {
            "event": "command_sample",
            "stamp_ns": 2_350_000_000,
            "stream": "smoothed_command",
            "linear_mps": speed - 0.05,
            "angular_rps": 0.0,
        },
        {
            "event": "odom_sample",
            "stamp_ns": 2_350_000_000,
            "stream": "wheel_odom",
            "x_m": 1.0,
            "y_m": 0.0,
            "linear_mps": speed - 0.05,
            "angular_rps": 0.0,
        },
        {
            "event": "odom_sample",
            "stamp_ns": 2_350_000_000,
            "stream": "fused_odom",
            "x_m": 1.0,
            "y_m": 0.0,
            "linear_mps": speed - 0.05,
            "angular_rps": 0.0,
        },
        {
            "event": "odom_sample",
            "stamp_ns": 2_600_000_000,
            "stream": "wheel_odom",
            "x_m": 1.06,
            "y_m": 0.0,
            "linear_mps": 0.0,
            "angular_rps": 0.0,
        },
        {
            "event": "odom_sample",
            "stamp_ns": 2_600_000_000,
            "stream": "fused_odom",
            "x_m": 1.06,
            "y_m": 0.0,
            "linear_mps": 0.0,
            "angular_rps": 0.0,
        },
        {
            "event": "health_transition",
            "stamp_ns": 2_700_000_000,
            "state": "HEALTHY",
            "reasons": [],
        },
        {
            "event": "trial_finished",
            "stamp_ns": 3 * NS,
            "outcome": "stop",
        },
    ]


class ResponseTrialAuditTest(unittest.TestCase):
    def test_recorder_requires_one_explicit_clock_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial clock"):
            ResponseTrialRecorder().record("trial_started")

        recorder = ResponseTrialRecorder(clock=lambda: 123)
        recorder.record("health_transition", state="HEALTHY", reasons=[])
        self.assertEqual(recorder.document()["events"][0]["stamp_ns"], 123)

    def test_physical_matrix_contains_no_claimed_evidence(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "obstacle_response_physical_trial_manifest_v1.json"
        )
        manifest = json.loads(manifest_path.read_bytes())

        self.assertEqual(manifest["evidence_state"], "PLANNED_NO_EVIDENCE")
        self.assertTrue(manifest["requires_ticket_7_terminal_decision"])
        self.assertTrue(manifest["requires_safety_policy_approval"])
        self.assertTrue(manifest["trials"])
        self.assertTrue(
            all(
                trial["status"] == "PLANNED_NO_EVIDENCE" for trial in manifest["trials"]
            )
        )

    def test_recorder_preserves_auditable_stream_events(self) -> None:
        recorder = ResponseTrialRecorder()
        recorder.record(
            "command_sample",
            stamp_ns=20,
            stream="smoothed_command",
            linear_mps=0.2,
            angular_rps=0.1,
        )
        recorder.record(
            "confirmed_hazard",
            stamp_ns=10,
            observation_stamp_ns=5,
            confirmed_stamp_ns=8,
            hazard_track_id=3,
            detection_distance_m=0.7,
        )

        document = recorder.document()

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["events"][0]["event"], "confirmed_hazard")
        self.assertEqual(document["events"][1]["stream"], "smoothed_command")

    def test_audit_derives_latency_and_stopping_envelope(self) -> None:
        report = evaluate_response_trial(_trial_events())

        self.assertEqual(report["evidence_state"], "MEASURED")
        self.assertEqual(report["outcome"], "stop")
        self.assertEqual(report["hazard_track_id"], 7)
        self.assertEqual(report["camera_to_response_ms"], 250.0)
        self.assertEqual(report["camera_to_response_start_ms"], 300.0)
        self.assertEqual(report["braking_start_stamp_ns"], 2_350_000_000)
        self.assertAlmostEqual(report["stopping_distance_m"], 0.06)
        self.assertEqual(report["health_transitions"][0]["state"], "HEALTHY")

    def test_speed_above_profile_is_rejected_not_validated(self) -> None:
        events = _trial_events(speed=0.31)
        report = evaluate_response_trial(events)

        self.assertEqual(report["evidence_state"], "REJECTED")
        self.assertIn("profile_speed_exceeded", report["rejection_reasons"])

    def test_wrong_profile_is_rejected_not_validated(self) -> None:
        events = _trial_events()
        events[0]["profile_id"] = "future-unvalidated-profile"

        report = evaluate_response_trial(events)

        self.assertEqual(report["evidence_state"], "REJECTED")
        self.assertIn("profile_id_mismatch", report["rejection_reasons"])

    def test_impossible_timestamp_order_is_rejected(self) -> None:
        events = _trial_events()
        next(
            event for event in events if event["event"] == "unified_response_received"
        )["stamp_ns"] = 1_900_000_000

        report = evaluate_response_trial(events)

        self.assertEqual(report["evidence_state"], "REJECTED")
        self.assertIn("timestamp_order_invalid", report["rejection_reasons"])

    def test_missing_physical_evidence_is_reported_not_invented(self) -> None:
        report = evaluate_response_trial(_trial_events()[:1])

        self.assertEqual(report["evidence_state"], "INCOMPLETE")
        self.assertIn("missing:confirmed_hazard", report["rejection_reasons"])
        self.assertIsNone(report["stopping_distance_m"])

    def test_named_events_without_required_evidence_remain_incomplete(self) -> None:
        events = _trial_events()
        confirmed = next(
            event for event in events if event["event"] == "confirmed_hazard"
        )
        del confirmed["observation_stamp_ns"]
        del confirmed["detection_distance_m"]
        finished = next(event for event in events if event["event"] == "trial_finished")
        del finished["outcome"]
        events = [
            event
            for event in events
            if event["event"] != "health_transition"
            and not (
                event["event"] == "command_sample" and event["stream"] == "command"
            )
        ]

        report = evaluate_response_trial(events)

        self.assertEqual(report["evidence_state"], "INCOMPLETE")
        self.assertIn(
            "missing:confirmed_hazard.observation_stamp_ns",
            report["rejection_reasons"],
        )
        self.assertIn("missing:command", report["rejection_reasons"])
        self.assertIn("missing:health_transition", report["rejection_reasons"])
        self.assertIn("missing:trial_finished.outcome", report["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
