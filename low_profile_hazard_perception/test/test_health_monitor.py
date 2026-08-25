import unittest

from low_profile_hazard_perception.health import (
    CameraInfoObservation,
    HealthMonitor,
    HealthState,
    ImageObservation,
    OdomObservation,
    Stream,
    Transform,
    TransformBatchObservation,
    geometric_projection_support_reason,
)


class DeterministicReplayHealthTests(unittest.TestCase):
    def test_canonical_replay_result_uses_sensor_time_not_receive_time(
        self,
    ) -> None:
        first = HealthMonitor(expected_width=640, expected_height=360)
        second = HealthMonitor(expected_width=640, expected_height=360)

        for index in range(3):
            stamp_ns = 1_000_000_000 + index * 100_000_000
            first.observe_image(
                Stream.COLOR_IMAGE,
                ImageObservation(
                    sensor_stamp_ns=stamp_ns,
                    receive_time_ns=10_000_000 + index * 7_000_000,
                    frame_id="camera_1_color_optical_frame",
                    width=640,
                    height=360,
                    step=640 * 3,
                    encoding="rgb8",
                    data_size=640 * 360 * 3,
                ),
            )
            second.observe_image(
                Stream.COLOR_IMAGE,
                ImageObservation(
                    sensor_stamp_ns=stamp_ns,
                    receive_time_ns=90_000_000 + index * 19_000_000,
                    frame_id="camera_1_color_optical_frame",
                    width=640,
                    height=360,
                    step=640 * 3,
                    encoding="rgb8",
                    data_size=640 * 360 * 3,
                ),
            )

        first_snapshot = first.snapshot(
            sensor_now_ns=1_300_000_000, receive_now_ns=40_000_000
        )
        second_snapshot = second.snapshot(
            sensor_now_ns=1_300_000_000, receive_now_ns=160_000_000
        )

        self.assertEqual(
            first_snapshot.canonical_replay_result(),
            second_snapshot.canonical_replay_result(),
        )
        self.assertEqual(
            first_snapshot.streams[Stream.COLOR_IMAGE].sensor_stamp_age_ms,
            100.0,
        )
        self.assertNotEqual(
            first_snapshot.streams[Stream.COLOR_IMAGE].receive_age_ms,
            second_snapshot.streams[Stream.COLOR_IMAGE].receive_age_ms,
        )


class InputContractTests(unittest.TestCase):
    def _observe_valid_image_pair(
        self, monitor: HealthMonitor, *, stamp_ns: int, receive_time_ns: int
    ) -> None:
        for stream, encoding, bytes_per_pixel in (
            (Stream.COLOR_IMAGE, "rgb8", 3),
            (Stream.DEPTH_IMAGE, "16UC1", 2),
        ):
            monitor.observe_image(
                stream,
                ImageObservation(
                    sensor_stamp_ns=stamp_ns,
                    receive_time_ns=receive_time_ns,
                    frame_id="camera_1_color_optical_frame",
                    width=640,
                    height=360,
                    step=640 * bytes_per_pixel,
                    encoding=encoding,
                    data_size=640 * 360 * bytes_per_pixel,
                ),
            )
        for stream in (
            Stream.COLOR_CAMERA_INFO,
            Stream.DEPTH_CAMERA_INFO,
        ):
            monitor.observe_camera_info(
                stream,
                CameraInfoObservation(
                    sensor_stamp_ns=stamp_ns,
                    receive_time_ns=receive_time_ns,
                    frame_id="camera_1_color_optical_frame",
                    width=640,
                    height=360,
                    fx=455.0,
                    fy=455.0,
                    cx=320.0,
                    cy=180.0,
                ),
            )

    def test_delivered_profile_and_camera_info_are_reported(self) -> None:
        monitor = HealthMonitor(expected_width=640, expected_height=360)
        monitor.observe_image(
            Stream.DEPTH_IMAGE,
            ImageObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                step=640 * 2,
                encoding="16UC1",
                data_size=640 * 360 * 2,
            ),
        )
        monitor.record_processing_complete(
            Stream.DEPTH_IMAGE,
            receive_time_ns=10_000_000,
            processing_complete_time_ns=16_000_000,
        )
        monitor.observe_camera_info(
            Stream.DEPTH_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=11_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                fx=455.0,
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )

        snapshot = monitor.snapshot(
            sensor_now_ns=1_020_000_000, receive_now_ns=20_000_000
        )
        depth = snapshot.streams[Stream.DEPTH_IMAGE]

        self.assertEqual(depth.profile, "640x360")
        self.assertEqual(depth.encoding, "16UC1")
        self.assertEqual(depth.frame_id, "camera_1_color_optical_frame")
        self.assertEqual(depth.processing_latency_ms, 6.0)
        self.assertTrue(snapshot.camera_info_consistency["depth"])

    def test_projection_support_reports_missing_and_stale_prerequisites(
        self,
    ) -> None:
        missing = HealthMonitor(expected_width=640, expected_height=360)
        missing_snapshot = missing.snapshot(
            sensor_now_ns=1_000_000_000,
            receive_now_ns=10_000_000,
        )
        self.assertEqual(
            geometric_projection_support_reason(missing_snapshot),
            "camera_info:missing",
        )
        missing.observe_camera_info(
            Stream.DEPTH_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                fx=455.0,
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )
        self.assertEqual(
            geometric_projection_support_reason(
                missing.snapshot(
                    sensor_now_ns=1_000_000_000,
                    receive_now_ns=10_000_000,
                )
            ),
            "tf:missing",
        )

        stale = HealthMonitor(expected_width=640, expected_height=360)
        stale.observe_camera_info(
            Stream.DEPTH_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                fx=455.0,
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )
        stale.observe_odom(
            OdomObservation(
                sensor_stamp_ns=1_600_000_000,
                receive_time_ns=20_000_000,
                frame_id="odom",
                child_frame_id="base_footprint",
            )
        )
        transform = Transform(
            parent_frame_id="base_footprint",
            child_frame_id="camera_1_color_optical_frame",
            translation=(0.33, 0.0, 0.15),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for stream in (Stream.TF, Stream.TF_STATIC):
            stale.observe_transforms(
                stream,
                TransformBatchObservation(
                    sensor_stamp_ns=1_600_000_000,
                    receive_time_ns=20_000_000,
                    transforms=(transform,),
                    required_chain_available=True,
                ),
            )
        stale_snapshot = stale.snapshot(
            sensor_now_ns=1_600_000_000,
            receive_now_ns=20_000_000,
        )

        self.assertEqual(
            geometric_projection_support_reason(stale_snapshot),
            "camera_info:sensor_stale",
        )

        stale_tf = HealthMonitor(expected_width=640, expected_height=360)
        stale_tf.observe_image(
            Stream.DEPTH_IMAGE,
            ImageObservation(
                sensor_stamp_ns=1_600_000_000,
                receive_time_ns=20_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                step=1280,
                encoding="16UC1",
                data_size=640 * 360 * 2,
            ),
        )
        stale_tf.observe_camera_info(
            Stream.DEPTH_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_600_000_000,
                receive_time_ns=20_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                fx=455.0,
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )
        stale_tf.observe_odom(
            OdomObservation(
                sensor_stamp_ns=1_600_000_000,
                receive_time_ns=20_000_000,
                frame_id="odom",
                child_frame_id="base_footprint",
            )
        )
        for stream in (Stream.TF, Stream.TF_STATIC):
            stale_tf.observe_transforms(
                stream,
                TransformBatchObservation(
                    sensor_stamp_ns=1_000_000_000,
                    receive_time_ns=20_000_000,
                    transforms=(transform,),
                    required_chain_available=True,
                ),
            )
        self.assertEqual(
            geometric_projection_support_reason(
                stale_tf.snapshot(
                    sensor_now_ns=1_600_000_000,
                    receive_now_ns=20_000_000,
                )
            ),
            "tf:sensor_stale",
        )

        for stream in (Stream.TF, Stream.TF_STATIC):
            stale_tf.observe_transforms(
                stream,
                TransformBatchObservation(
                    sensor_stamp_ns=1_600_000_000,
                    receive_time_ns=20_000_000,
                    transforms=(transform,),
                    required_chain_available=True,
                ),
            )
        stale_tf.observe_image(
            Stream.DEPTH_IMAGE,
            ImageObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=20_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                step=1280,
                encoding="16UC1",
                data_size=640 * 360 * 2,
            ),
        )
        self.assertEqual(
            geometric_projection_support_reason(
                stale_tf.snapshot(
                    sensor_now_ns=1_600_000_000,
                    receive_now_ns=20_000_000,
                )
            ),
            "depth:sensor_stale",
        )

    def test_invalid_dimensions_stride_encoding_and_camera_info_are_invalid(
        self,
    ) -> None:
        invalid_images = (
            ("dimensions", {"height": 400}),
            ("stride", {"step": 100}),
            ("encoding", {"encoding": "32FC1"}),
            ("data size", {"data_size": 100}),
        )
        defaults = {
            "sensor_stamp_ns": 1_000_000_000,
            "receive_time_ns": 10_000_000,
            "frame_id": "camera_1_color_optical_frame",
            "width": 640,
            "height": 360,
            "step": 640 * 2,
            "encoding": "16UC1",
            "data_size": 640 * 360 * 2,
        }
        for label, override in invalid_images:
            with self.subTest(label=label):
                monitor = HealthMonitor(
                    expected_width=640, expected_height=360
                )
                monitor.observe_image(
                    Stream.DEPTH_IMAGE,
                    ImageObservation(**(defaults | override)),
                )
                snapshot = monitor.snapshot(
                    sensor_now_ns=1_010_000_000,
                    receive_now_ns=20_000_000,
                )
                self.assertEqual(snapshot.state, HealthState.INVALID)
                self.assertIn(Stream.DEPTH_IMAGE, snapshot.invalid_streams)

        monitor = HealthMonitor(expected_width=640, expected_height=360)
        monitor.observe_image(Stream.DEPTH_IMAGE, ImageObservation(**defaults))
        monitor.observe_camera_info(
            Stream.DEPTH_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=11_000_000,
                frame_id="another_optical_frame",
                width=640,
                height=360,
                fx=455.0,
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )
        snapshot = monitor.snapshot(
            sensor_now_ns=1_010_000_000, receive_now_ns=20_000_000
        )
        self.assertEqual(snapshot.state, HealthState.INVALID)
        self.assertFalse(snapshot.camera_info_consistency["depth"])

    def test_nonfinite_intrinsics_and_unexpected_camera_frame_are_invalid(
        self,
    ) -> None:
        monitor = HealthMonitor(expected_width=640, expected_height=360)
        monitor.observe_camera_info(
            Stream.COLOR_CAMERA_INFO,
            CameraInfoObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                fx=float("nan"),
                fy=455.0,
                cx=320.0,
                cy=180.0,
            ),
        )
        monitor.observe_image(
            Stream.COLOR_IMAGE,
            ImageObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="unexpected_camera_frame",
                width=640,
                height=360,
                step=640 * 3,
                encoding="rgb8",
                data_size=640 * 360 * 3,
            ),
        )

        snapshot = monitor.snapshot(
            sensor_now_ns=1_010_000_000,
            receive_now_ns=20_000_000,
        )

        self.assertEqual(snapshot.state, HealthState.INVALID)
        self.assertIn(Stream.COLOR_IMAGE, snapshot.invalid_streams)
        self.assertIn(Stream.COLOR_CAMERA_INFO, snapshot.invalid_streams)

    def test_missing_and_stale_measurements_are_not_conflated(self) -> None:
        monitor = HealthMonitor(
            expected_width=640,
            expected_height=360,
            sensor_stale_after_ns=200_000_000,
            receive_stale_after_ns=100_000_000,
        )
        monitor.observe_image(
            Stream.COLOR_IMAGE,
            ImageObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=900_000_000,
                frame_id="camera_1_color_optical_frame",
                width=640,
                height=360,
                step=640 * 3,
                encoding="rgb8",
                data_size=640 * 360 * 3,
            ),
        )
        monitor.record_queue_drops(Stream.COLOR_IMAGE, 2)
        monitor.record_transport_drops(Stream.COLOR_IMAGE, 1)

        snapshot = monitor.snapshot(
            sensor_now_ns=1_300_000_000,
            receive_now_ns=1_050_000_000,
        )

        self.assertEqual(snapshot.state, HealthState.DEGRADED)
        self.assertIn(Stream.DEPTH_IMAGE, snapshot.missing_streams)
        self.assertIn(Stream.COLOR_IMAGE, snapshot.stale_sensor_streams)
        self.assertIn(Stream.COLOR_IMAGE, snapshot.stale_receive_streams)
        self.assertEqual(snapshot.streams[Stream.COLOR_IMAGE].queue_drops, 2)
        self.assertEqual(
            snapshot.streams[Stream.COLOR_IMAGE].transport_drops, 1
        )
        self.assertNotIn(Stream.COLOR_IMAGE, snapshot.missing_streams)

    def test_all_required_independent_streams_can_be_healthy(self) -> None:
        monitor = HealthMonitor(expected_width=640, expected_height=360)
        self._observe_valid_image_pair(
            monitor, stamp_ns=1_000_000_000, receive_time_ns=10_000_000
        )
        monitor.observe_odom(
            OdomObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                frame_id="odom",
                child_frame_id="base_footprint",
            )
        )
        transform = Transform(
            parent_frame_id="base_footprint",
            child_frame_id="camera_1_color_optical_frame",
            translation=(0.33, 0.0, 0.15),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        monitor.observe_transforms(
            Stream.TF,
            TransformBatchObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                transforms=(transform,),
                required_chain_available=True,
            ),
        )
        monitor.observe_transforms(
            Stream.TF_STATIC,
            TransformBatchObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                transforms=(transform,),
                required_chain_available=True,
            ),
        )

        snapshot = monitor.snapshot(
            sensor_now_ns=1_020_000_000,
            receive_now_ns=20_000_000,
        )

        self.assertEqual(snapshot.state, HealthState.HEALTHY)
        self.assertEqual(snapshot.missing_streams, ())

    def test_invalid_tf_is_explicitly_invalid(self) -> None:
        monitor = HealthMonitor(expected_width=640, expected_height=360)
        monitor.observe_transforms(
            Stream.TF,
            TransformBatchObservation(
                sensor_stamp_ns=1_000_000_000,
                receive_time_ns=10_000_000,
                transforms=(
                    Transform(
                        parent_frame_id="base_footprint",
                        child_frame_id="camera_1_color_optical_frame",
                        translation=(0.33, 0.0, 0.15),
                        rotation=(0.0, 0.0, 0.0, 0.0),
                    ),
                ),
                required_chain_available=False,
            ),
        )

        snapshot = monitor.snapshot(
            sensor_now_ns=1_010_000_000,
            receive_now_ns=20_000_000,
        )

        self.assertEqual(snapshot.state, HealthState.INVALID)
        self.assertIn(Stream.TF, snapshot.invalid_streams)


if __name__ == "__main__":
    unittest.main()
