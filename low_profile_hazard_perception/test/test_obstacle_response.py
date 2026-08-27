import struct
import unittest
from pathlib import Path

from low_profile_hazard_perception.obstacle_response import (
    ConfirmedCloudView,
    HealthState,
    ObstacleResponseConfig,
    PerceptionHealth,
    RecordingObstacleResponsePort,
    ResponseSourceMode,
    UnifiedObstacleResponseBridge,
    decode_confirmed_cloud,
    perception_health_from_values,
)


NS = 1_000_000_000


def _health(
    *,
    state: HealthState = HealthState.HEALTHY,
    received_ns: int = 10 * NS,
    heartbeat_ns: int = 10 * NS,
    profile_id: str = "dcw2-home-640x360-v1",
    binding_state: str = "BOUND",
    maximum_speed_mps: float = 0.3,
    observed_speed_mps: float = 0.3,
) -> PerceptionHealth:
    return PerceptionHealth(
        state=state,
        received_monotonic_ns=received_ns,
        heartbeat_stamp_ns=heartbeat_ns,
        profile_id=profile_id,
        profile_binding_state=binding_state,
        profile_maximum_speed_mps=maximum_speed_mps,
        observed_speed_mps=observed_speed_mps,
    )


def _cloud(
    *,
    stamp_ns: int = 9 * NS,
    track_id: int = 7,
    empty: bool = False,
) -> ConfirmedCloudView:
    point_step = 40
    data = b""
    width = 0
    if not empty:
        seconds, nanoseconds = divmod(stamp_ns, NS)
        data = struct.pack(
            "<ffffiIB3xfII",
            1.2,
            -0.1,
            0.02,
            0.01,
            seconds,
            nanoseconds,
            1,
            100.0,
            0,
            track_id,
        )
        width = 1
    return ConfirmedCloudView(
        frame_id="odom",
        header_stamp_ns=stamp_ns,
        width=width,
        height=1,
        point_step=point_step,
        row_step=point_step * width,
        data=data,
        fields={
            "x": (0, "float32"),
            "y": (4, "float32"),
            "z": (8, "float32"),
            "observation_stamp_sec": (16, "int32"),
            "observation_stamp_nanosec": (20, "uint32"),
            "cloud_group_index": (32, "uint32"),
            "hazard_track_id": (36, "uint32"),
        },
    )


class UnifiedObstacleResponseBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.port = RecordingObstacleResponsePort()
        self.bridge = UnifiedObstacleResponseBridge(
            ObstacleResponseConfig(
                expected_profile_id="dcw2-home-640x360-v1",
                maximum_speed_mps=0.3,
                health_timeout_ns=600_000_000,
                maximum_observation_age_ns=2_500_000_000,
            ),
            self.port,
        )

    def test_confirmed_only_snapshot_reaches_unified_response(self) -> None:
        self.bridge.consume_health(_health())

        accepted = self.bridge.consume_cloud(
            decode_confirmed_cloud(_cloud()),
            sensor_now_ns=10 * NS,
            received_monotonic_ns=10 * NS,
        )

        self.assertTrue(accepted)
        self.assertEqual(self.port.snapshots[-1].hazards[0].hazard_track_id, 7)
        self.assertEqual(self.port.snapshots[-1].observation_stamp_ns, 9 * NS)
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.ACTIVE)

    def test_cloud_first_startup_is_buffered_and_stale_health_never_clears(
        self,
    ) -> None:
        snapshot = decode_confirmed_cloud(_cloud())
        self.assertFalse(
            self.bridge.consume_cloud(
                snapshot,
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
        )
        self.assertEqual(self.port.snapshots, [])

        self.bridge.consume_health(_health())
        self.assertEqual(len(self.port.snapshots), 1)
        self.assertEqual(
            self.port.snapshots[0].hazards[0].hazard_track_id,
            snapshot.hazards[0].hazard_track_id,
        )
        self.bridge.tick(received_monotonic_ns=11 * NS)
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.BLOCKED)
        self.assertEqual(len(self.port.snapshots), 1)

        self.assertFalse(
            self.bridge.consume_cloud(
                decode_confirmed_cloud(_cloud(stamp_ns=11 * NS, empty=True)),
                sensor_now_ns=11 * NS,
                received_monotonic_ns=11 * NS,
            )
        )
        self.assertEqual(len(self.port.snapshots), 1)

    def test_invalid_holds_and_degraded_accepts_confirmed_but_not_clear(self) -> None:
        self.bridge.consume_health(_health(state=HealthState.INVALID))
        self.assertFalse(
            self.bridge.consume_cloud(
                decode_confirmed_cloud(_cloud()),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
        )
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.BLOCKED)

        self.bridge.consume_health(_health(state=HealthState.DEGRADED))
        self.assertEqual(len(self.port.snapshots), 1)
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.DEGRADED)
        self.assertFalse(
            self.bridge.consume_cloud(
                decode_confirmed_cloud(_cloud(stamp_ns=10 * NS, empty=True)),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
        )
        self.assertEqual(len(self.port.snapshots), 1)

    def test_explicit_empty_cloud_clears_only_while_healthy(self) -> None:
        self.bridge.consume_health(_health())
        self.bridge.consume_cloud(
            decode_confirmed_cloud(_cloud()),
            sensor_now_ns=10 * NS,
            received_monotonic_ns=10 * NS,
        )
        self.bridge.tick(received_monotonic_ns=10_500_000_000)
        self.assertEqual(len(self.port.snapshots), 1)

        self.assertTrue(
            self.bridge.consume_cloud(
                decode_confirmed_cloud(_cloud(stamp_ns=10 * NS, empty=True)),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10_500_000_000,
            )
        )
        self.assertTrue(self.port.snapshots[-1].explicit_empty)

    def test_health_clock_regression_starts_a_new_source_generation(self) -> None:
        self.bridge.consume_health(_health())
        first_generation = self.port.statuses[-1].source_generation
        self.bridge.consume_health(_health(received_ns=11 * NS, heartbeat_ns=5 * NS))

        self.assertEqual(
            self.port.statuses[-1].source_generation,
            first_generation + 1,
        )
        self.assertTrue(self.port.statuses[-1].awaiting_fresh_snapshot)
        self.bridge.consume_cloud(
            decode_confirmed_cloud(_cloud(stamp_ns=10 * NS, track_id=0)),
            sensor_now_ns=11 * NS,
            received_monotonic_ns=11 * NS,
        )
        self.assertFalse(self.port.statuses[-1].awaiting_fresh_snapshot)

    def test_health_recovery_after_liveness_loss_starts_a_new_generation(self) -> None:
        self.bridge.consume_health(_health())
        first_generation = self.port.statuses[-1].source_generation
        self.bridge.tick(received_monotonic_ns=11 * NS)

        self.bridge.consume_health(_health(received_ns=11 * NS, heartbeat_ns=11 * NS))

        self.assertEqual(
            self.port.statuses[-1].source_generation,
            first_generation + 1,
        )
        self.assertTrue(self.port.statuses[-1].awaiting_fresh_snapshot)

    def test_provider_diagnostics_do_not_change_the_response(self) -> None:
        outputs = []
        common = {
            "profile.id": "dcw2-home-640x360-v1",
            "profile.binding_state": "BOUND",
            "profile.maximum_speed_mps": "0.3",
            "profile.latest_observed_speed_mps": "0.3",
        }
        for diagnostics in (
            {
                "cable.provider": "training_free_thin_line",
                "resource.npu_state": "disabled_rule_profile",
                "candidate.count": "999",
            },
            {
                "cable.provider": "future_rknn_provider",
                "resource.npu_state": "unavailable",
                "candidate.count": "0",
            },
        ):
            port = RecordingObstacleResponsePort()
            bridge = UnifiedObstacleResponseBridge(self.bridge.config, port)
            bridge.consume_health(
                perception_health_from_values(
                    state="HEALTHY",
                    values={**common, **diagnostics},
                    received_monotonic_ns=10 * NS,
                    heartbeat_stamp_ns=10 * NS,
                )
            )
            bridge.consume_cloud(
                decode_confirmed_cloud(_cloud()),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
            outputs.append((port.statuses, port.snapshots))

        self.assertEqual(outputs[0], outputs[1])

    def test_health_parser_uses_an_allow_list_not_provider_diagnostics(self) -> None:
        common = {
            "profile.id": "dcw2-home-640x360-v1",
            "profile.binding_state": "BOUND",
            "profile.maximum_speed_mps": "0.3",
            "profile.latest_observed_speed_mps": "0.3",
        }
        rule = perception_health_from_values(
            state="DEGRADED",
            values={
                **common,
                "cable.provider": "training_free_thin_line",
                "resource.npu_state": "disabled_rule_profile",
            },
            received_monotonic_ns=10 * NS,
            heartbeat_stamp_ns=10 * NS,
        )
        npu = perception_health_from_values(
            state="DEGRADED",
            values={
                **common,
                "cable.provider": "future_npu",
                "resource.npu_state": "failed",
            },
            received_monotonic_ns=10 * NS,
            heartbeat_stamp_ns=10 * NS,
        )

        self.assertEqual(rule, npu)
        self.assertFalse(hasattr(rule, "diagnostics"))

    def test_npu_unavailable_does_not_hide_degraded_geometry_output(self) -> None:
        self.bridge.consume_health(
            perception_health_from_values(
                state="DEGRADED",
                values={
                    "profile.id": "dcw2-home-640x360-v1",
                    "profile.binding_state": "BOUND",
                    "profile.maximum_speed_mps": "0.3",
                    "profile.latest_observed_speed_mps": "0.3",
                    "cable.provider": "future_npu",
                    "resource.npu_state": "unavailable",
                },
                received_monotonic_ns=10 * NS,
                heartbeat_stamp_ns=10 * NS,
            )
        )

        accepted = self.bridge.consume_cloud(
            decode_confirmed_cloud(_cloud()),
            sensor_now_ns=10 * NS,
            received_monotonic_ns=10 * NS,
        )

        self.assertTrue(accepted)
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.DEGRADED)
        self.assertEqual(len(self.port.snapshots), 1)

    def test_profile_speed_and_observation_age_are_hard_bounds(self) -> None:
        self.bridge.consume_health(_health(observed_speed_mps=0.301))
        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.BLOCKED)
        self.assertFalse(
            self.bridge.consume_cloud(
                decode_confirmed_cloud(_cloud()),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
        )

        port = RecordingObstacleResponsePort()
        bridge = UnifiedObstacleResponseBridge(self.bridge.config, port)
        bridge.consume_health(_health())
        self.assertFalse(
            bridge.consume_cloud(
                decode_confirmed_cloud(_cloud(stamp_ns=7 * NS)),
                sensor_now_ns=10 * NS,
                received_monotonic_ns=10 * NS,
            )
        )
        self.assertEqual(port.statuses[-1].mode, ResponseSourceMode.BLOCKED)

    def test_cloud_contract_requires_odom_and_per_point_observation_stamps(
        self,
    ) -> None:
        wrong_frame = _cloud()
        wrong_frame = ConfirmedCloudView(**{**wrong_frame.__dict__, "frame_id": "map"})
        with self.assertRaisesRegex(ValueError, "odom"):
            decode_confirmed_cloud(wrong_frame)

        fields = dict(_cloud().fields)
        del fields["hazard_track_id"]
        missing_track = ConfirmedCloudView(**{**_cloud().__dict__, "fields": fields})
        with self.assertRaisesRegex(ValueError, "hazard_track_id"):
            decode_confirmed_cloud(missing_track)

    def test_versioned_preintegration_config_binds_the_current_envelope(self) -> None:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "obstacle_response_adapter_v1.json"
        )

        config = ObstacleResponseConfig.load(config_path)

        self.assertEqual(config.expected_profile_id, "dcw2-home-640x360-v1")
        self.assertEqual(config.maximum_speed_mps, 0.3)
        self.assertEqual(config.health_timeout_ns, 600_000_000)
        self.assertEqual(config.maximum_observation_age_ns, 2_500_000_000)

    def test_malformed_operational_input_blocks_without_clearing(self) -> None:
        self.bridge.consume_health(_health())
        self.bridge.consume_cloud(
            decode_confirmed_cloud(_cloud()),
            sensor_now_ns=10 * NS,
            received_monotonic_ns=10 * NS,
        )

        self.bridge.reject_operational_input("confirmed_cloud_invalid")

        self.assertEqual(self.port.statuses[-1].mode, ResponseSourceMode.BLOCKED)
        self.assertEqual(len(self.port.snapshots), 1)


if __name__ == "__main__":
    unittest.main()
