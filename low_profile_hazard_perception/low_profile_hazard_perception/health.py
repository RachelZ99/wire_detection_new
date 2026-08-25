"""Deterministic input-health aggregation.

The monitor deliberately receives sensor-clock and monotonic receive times.
Keeping those clocks explicit prevents transport delay from being mistaken for
host-side queueing or staleness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any

from .replay_result import is_timing_field


class Stream(str, Enum):
    COLOR_IMAGE = "color_image"
    COLOR_CAMERA_INFO = "color_camera_info"
    DEPTH_IMAGE = "depth_image"
    DEPTH_CAMERA_INFO = "depth_camera_info"
    ODOM = "odom"
    TF = "tf"
    TF_STATIC = "tf_static"


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ImageObservation:
    sensor_stamp_ns: int
    receive_time_ns: int
    frame_id: str
    width: int
    height: int
    step: int
    encoding: str
    data_size: int


@dataclass(frozen=True)
class CameraInfoObservation:
    sensor_stamp_ns: int
    receive_time_ns: int
    frame_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class OdomObservation:
    sensor_stamp_ns: int
    receive_time_ns: int
    frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class Transform:
    parent_frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class TransformBatchObservation:
    sensor_stamp_ns: int
    receive_time_ns: int
    transforms: tuple[Transform, ...]
    required_chain_available: bool
    input_error: str = ""


def validate_transform_batch(
    stream: Stream, observation: TransformBatchObservation
) -> str:
    """Validate a TF batch before an adapter gives it to tf2."""
    if stream not in (Stream.TF, Stream.TF_STATIC):
        return f"{stream.value} is not a TF stream"
    if observation.input_error:
        return observation.input_error
    if observation.sensor_stamp_ns <= 0 and stream is Stream.TF:
        return "sensor stamp must be positive"
    if not observation.transforms:
        return "TF message has no transforms"
    for transform in observation.transforms:
        reason = _validate_transform(transform)
        if reason:
            return reason
    return ""


def _validate_transform(transform: Transform) -> str:
    if not transform.parent_frame_id or not transform.child_frame_id:
        return "TF parent and child frame IDs are required"
    if transform.parent_frame_id == transform.child_frame_id:
        return "TF parent and child frame IDs must differ"
    values = (*transform.translation, *transform.rotation)
    if not all(isfinite(value) for value in values):
        return "TF contains a non-finite value"
    norm = sqrt(sum(value * value for value in transform.rotation))
    if norm < 1e-6:
        return "TF rotation quaternion has zero norm"
    if abs(norm - 1.0) > 1e-3:
        return f"TF rotation quaternion norm is {norm:.6f}, expected 1"
    return ""


@dataclass(frozen=True)
class StreamHealth:
    delivered_count: int
    valid_count: int
    invalid_count: int
    queue_drops: int
    transport_drops: int
    sensor_stamp_age_ms: float | None
    receive_age_ms: float | None
    processing_latency_ms: float | None
    approximate_rate_hz: float | None
    frame_id: str
    profile: str
    encoding: str
    valid: bool
    reason: str


@dataclass(frozen=True)
class HealthSnapshot:
    state: HealthState
    streams: dict[Stream, StreamHealth]
    camera_info_consistency: dict[str, bool | None]
    missing_streams: tuple[Stream, ...]
    stale_sensor_streams: tuple[Stream, ...]
    stale_receive_streams: tuple[Stream, ...]
    invalid_streams: tuple[Stream, ...]
    tf_chain_available: bool | None
    reasons: tuple[str, ...]

    def canonical_replay_result(self) -> dict[str, Any]:
        """Return replay-stable fields, excluding host receive-time ages."""
        return {
            "state": self.state.value,
            "streams": {
                stream.value: {
                    key: value
                    for key, value in asdict(health).items()
                    if not is_timing_field(key)
                }
                for stream, health in sorted(
                    self.streams.items(), key=lambda item: item[0].value
                )
            },
            "camera_info_consistency": dict(
                sorted(self.camera_info_consistency.items())
            ),
            "missing_streams": [
                stream.value for stream in self.missing_streams
            ],
            "stale_sensor_streams": [
                stream.value for stream in self.stale_sensor_streams
            ],
            "invalid_streams": [
                stream.value for stream in self.invalid_streams
            ],
            "tf_chain_available": self.tf_chain_available,
            "reasons": list(self.reasons),
        }


def geometric_projection_support_reason(snapshot: HealthSnapshot) -> str:
    """Explain why depth evidence cannot be projected into ``odom``."""
    required = (
        (Stream.DEPTH_CAMERA_INFO, "camera_info"),
        (Stream.TF, "tf"),
        (Stream.TF_STATIC, "tf_static"),
        (Stream.ODOM, "odom"),
    )
    for stream, label in required:
        if stream in snapshot.missing_streams:
            return f"{label}:missing"
        if stream in snapshot.invalid_streams:
            return f"{label}:invalid"
        if stream in snapshot.stale_sensor_streams:
            return f"{label}:sensor_stale"
        if stream in snapshot.stale_receive_streams:
            return f"{label}:receive_stale"
    if snapshot.camera_info_consistency["depth"] is not True:
        return "camera_info:inconsistent"
    if snapshot.tf_chain_available is not True:
        return "tf:chain_unavailable"
    return ""


@dataclass
class _StreamRecord:
    delivered_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    queue_drops: int = 0
    transport_drops: int = 0
    first_sensor_stamp_ns: int | None = None
    last_sensor_stamp_ns: int | None = None
    last_receive_time_ns: int | None = None
    frame_id: str = ""
    width: int | None = None
    height: int | None = None
    encoding: str = ""
    valid: bool = False
    reason: str = "missing"
    processing_latency_ms: float | None = None
    observation: ImageObservation | CameraInfoObservation | None = None


class HealthMonitor:
    def __init__(
        self,
        *,
        expected_width: int,
        expected_height: int,
        expected_camera_frame: str = "camera_1_color_optical_frame",
        sensor_stale_after_ns: int = 500_000_000,
        receive_stale_after_ns: int = 500_000_000,
    ) -> None:
        self._expected_width = expected_width
        self._expected_height = expected_height
        self._expected_camera_frame = expected_camera_frame
        self._sensor_stale_after_ns = sensor_stale_after_ns
        self._receive_stale_after_ns = receive_stale_after_ns
        self._records = {stream: _StreamRecord() for stream in Stream}
        self._tf_chain_available: bool | None = None

    def observe_image(
        self, stream: Stream, observation: ImageObservation
    ) -> None:
        if stream not in (Stream.COLOR_IMAGE, Stream.DEPTH_IMAGE):
            raise ValueError(f"{stream.value} is not an image stream")
        expected_encoding = "rgb8" if stream is Stream.COLOR_IMAGE else "16UC1"
        bytes_per_pixel = 3 if stream is Stream.COLOR_IMAGE else 2
        reason = self._validate_common(observation)
        if not reason and observation.frame_id != self._expected_camera_frame:
            reason = (
                f"unexpected frame_id {observation.frame_id!r}; "
                f"expected {self._expected_camera_frame!r}"
            )
        if not reason and (
            observation.width != self._expected_width
            or observation.height != self._expected_height
        ):
            reason = (
                "unexpected profile "
                f"{observation.width}x{observation.height}; "
                f"expected {self._expected_width}x{self._expected_height}"
            )
        if not reason and observation.encoding != expected_encoding:
            reason = (
                f"unexpected encoding {observation.encoding}; "
                f"expected {expected_encoding}"
            )
        minimum_step = observation.width * bytes_per_pixel
        if not reason and observation.step < minimum_step:
            reason = (
                f"stride {observation.step} is smaller than {minimum_step}"
            )
        if (
            not reason
            and observation.data_size < observation.step * observation.height
        ):
            reason = "image data is smaller than stride times height"
        self._record(stream, observation, reason)

    def observe_camera_info(
        self, stream: Stream, observation: CameraInfoObservation
    ) -> None:
        if stream not in (
            Stream.COLOR_CAMERA_INFO,
            Stream.DEPTH_CAMERA_INFO,
        ):
            raise ValueError(f"{stream.value} is not a CameraInfo stream")
        reason = self._validate_common(observation)
        if not reason and observation.frame_id != self._expected_camera_frame:
            reason = (
                f"unexpected frame_id {observation.frame_id!r}; "
                f"expected {self._expected_camera_frame!r}"
            )
        if not reason and (
            observation.width != self._expected_width
            or observation.height != self._expected_height
        ):
            reason = (
                f"unexpected CameraInfo profile "
                f"{observation.width}x{observation.height}"
            )
        intrinsics = (
            observation.fx,
            observation.fy,
            observation.cx,
            observation.cy,
        )
        if not reason and not all(isfinite(value) for value in intrinsics):
            reason = "CameraInfo intrinsics must be finite"
        if not reason and (observation.fx <= 0.0 or observation.fy <= 0.0):
            reason = "CameraInfo focal lengths must be positive"
        if not reason and not (
            0.0 <= observation.cx < observation.width
            and 0.0 <= observation.cy < observation.height
        ):
            reason = "CameraInfo principal point is outside the image"
        self._record(stream, observation, reason)

    def observe_odom(self, observation: OdomObservation) -> None:
        reason = ""
        if observation.sensor_stamp_ns <= 0:
            reason = "sensor stamp must be positive"
        elif observation.frame_id != "odom":
            reason = (
                f"odom frame_id is {observation.frame_id!r}, expected 'odom'"
            )
        elif not observation.child_frame_id:
            reason = "odom child_frame_id is missing"
        self._record_generic(
            Stream.ODOM,
            sensor_stamp_ns=observation.sensor_stamp_ns,
            receive_time_ns=observation.receive_time_ns,
            frame_id=f"{observation.frame_id}->{observation.child_frame_id}",
            reason=reason,
        )

    def observe_transforms(
        self, stream: Stream, observation: TransformBatchObservation
    ) -> None:
        if stream not in (Stream.TF, Stream.TF_STATIC):
            raise ValueError(f"{stream.value} is not a TF stream")
        reason = validate_transform_batch(stream, observation)
        self._tf_chain_available = observation.required_chain_available
        frame_id = ",".join(
            f"{transform.parent_frame_id}->{transform.child_frame_id}"
            for transform in observation.transforms
        )
        self._record_generic(
            stream,
            sensor_stamp_ns=observation.sensor_stamp_ns,
            receive_time_ns=observation.receive_time_ns,
            frame_id=frame_id,
            reason=reason,
        )

    def snapshot(
        self, *, sensor_now_ns: int, receive_now_ns: int
    ) -> HealthSnapshot:
        streams: dict[Stream, StreamHealth] = {}
        for stream, record in self._records.items():
            rate = None
            if (
                record.delivered_count > 1
                and record.first_sensor_stamp_ns is not None
                and record.last_sensor_stamp_ns is not None
                and record.last_sensor_stamp_ns > record.first_sensor_stamp_ns
            ):
                rate = (
                    (record.delivered_count - 1)
                    * 1_000_000_000
                    / (
                        record.last_sensor_stamp_ns
                        - record.first_sensor_stamp_ns
                    )
                )
            sensor_age = self._age_ms(
                sensor_now_ns, record.last_sensor_stamp_ns
            )
            receive_age = self._age_ms(
                receive_now_ns, record.last_receive_time_ns
            )
            profile = ""
            if record.width is not None and record.height is not None:
                profile = f"{record.width}x{record.height}"
            streams[stream] = StreamHealth(
                delivered_count=record.delivered_count,
                valid_count=record.valid_count,
                invalid_count=record.invalid_count,
                queue_drops=record.queue_drops,
                transport_drops=record.transport_drops,
                sensor_stamp_age_ms=sensor_age,
                receive_age_ms=receive_age,
                processing_latency_ms=record.processing_latency_ms,
                approximate_rate_hz=rate,
                frame_id=record.frame_id,
                profile=profile,
                encoding=record.encoding,
                valid=record.valid,
                reason=record.reason,
            )
        consistency = {
            "color": self._camera_info_consistent(
                Stream.COLOR_IMAGE, Stream.COLOR_CAMERA_INFO
            ),
            "depth": self._camera_info_consistent(
                Stream.DEPTH_IMAGE, Stream.DEPTH_CAMERA_INFO
            ),
        }
        missing = tuple(
            stream for stream in Stream if streams[stream].delivered_count == 0
        )
        stale_sensor = tuple(
            stream
            for stream in Stream
            if stream is not Stream.TF_STATIC
            if streams[stream].sensor_stamp_age_ms is not None
            and streams[stream].sensor_stamp_age_ms
            > self._sensor_stale_after_ns / 1_000_000
        )
        stale_receive = tuple(
            stream
            for stream in Stream
            if stream is not Stream.TF_STATIC
            if streams[stream].receive_age_ms is not None
            and streams[stream].receive_age_ms
            > self._receive_stale_after_ns / 1_000_000
        )
        invalid = tuple(
            stream
            for stream in Stream
            if streams[stream].delivered_count > 0
            and not streams[stream].valid
        )
        inconsistent_camera_info = tuple(
            name
            for name, is_consistent in consistency.items()
            if is_consistent is False
        )
        tf_inputs_received = all(
            streams[stream].delivered_count > 0
            for stream in (Stream.TF, Stream.TF_STATIC)
        )
        tf_chain_invalid = (
            tf_inputs_received and self._tf_chain_available is False
        )
        queue_drops = sum(health.queue_drops for health in streams.values())
        transport_drops = sum(
            health.transport_drops for health in streams.values()
        )
        reasons = tuple(
            [*(f"invalid:{stream.value}" for stream in invalid)]
            + [
                *(
                    f"camera_info_inconsistent:{name}"
                    for name in inconsistent_camera_info
                )
            ]
            + (["tf_chain_unavailable"] if tf_chain_invalid else [])
            + [*(f"missing:{stream.value}" for stream in missing)]
            + [*(f"sensor_stale:{stream.value}" for stream in stale_sensor)]
            + [*(f"receive_stale:{stream.value}" for stream in stale_receive)]
            + ([f"queue_drops:{queue_drops}"] if queue_drops else [])
            + (
                [f"transport_drops:{transport_drops}"]
                if transport_drops
                else []
            )
        )
        if invalid or inconsistent_camera_info or tf_chain_invalid:
            state = HealthState.INVALID
        elif (
            missing
            or stale_sensor
            or stale_receive
            or queue_drops
            or transport_drops
        ):
            state = HealthState.DEGRADED
        else:
            state = HealthState.HEALTHY
        return HealthSnapshot(
            state=state,
            streams=streams,
            camera_info_consistency=consistency,
            missing_streams=missing,
            stale_sensor_streams=stale_sensor,
            stale_receive_streams=stale_receive,
            invalid_streams=invalid,
            tf_chain_available=self._tf_chain_available,
            reasons=reasons,
        )

    def record_queue_drops(self, stream: Stream, count: int) -> None:
        if count < 0:
            raise ValueError("queue drop count cannot be negative")
        self._records[stream].queue_drops += count

    def record_transport_drops(self, stream: Stream, count: int) -> None:
        if count < 0:
            raise ValueError("transport drop count cannot be negative")
        self._records[stream].transport_drops += count

    def record_processing_complete(
        self,
        stream: Stream,
        *,
        receive_time_ns: int,
        processing_complete_time_ns: int,
    ) -> None:
        if processing_complete_time_ns < receive_time_ns:
            raise ValueError("processing completion precedes receive time")
        self._records[stream].processing_latency_ms = (
            processing_complete_time_ns - receive_time_ns
        ) / 1_000_000

    @staticmethod
    def _age_ms(now_ns: int, then_ns: int | None) -> float | None:
        if then_ns is None:
            return None
        return (now_ns - then_ns) / 1_000_000

    @staticmethod
    def _validate_common(
        observation: ImageObservation | CameraInfoObservation,
    ) -> str:
        if observation.sensor_stamp_ns <= 0:
            return "sensor stamp must be positive"
        if not observation.frame_id:
            return "frame_id is missing"
        if observation.width <= 0 or observation.height <= 0:
            return "image dimensions must be positive"
        return ""

    def _record(
        self,
        stream: Stream,
        observation: ImageObservation | CameraInfoObservation,
        reason: str,
    ) -> None:
        record = self._records[stream]
        self._record_delivery(
            record,
            sensor_stamp_ns=observation.sensor_stamp_ns,
            receive_time_ns=observation.receive_time_ns,
            frame_id=observation.frame_id,
            reason=reason,
        )
        record.width = observation.width
        record.height = observation.height
        record.encoding = getattr(observation, "encoding", "")
        record.observation = observation

    def _record_generic(
        self,
        stream: Stream,
        *,
        sensor_stamp_ns: int,
        receive_time_ns: int,
        frame_id: str,
        reason: str,
    ) -> None:
        record = self._records[stream]
        self._record_delivery(
            record,
            sensor_stamp_ns=sensor_stamp_ns,
            receive_time_ns=receive_time_ns,
            frame_id=frame_id,
            reason=reason,
        )

    @staticmethod
    def _record_delivery(
        record: _StreamRecord,
        *,
        sensor_stamp_ns: int,
        receive_time_ns: int,
        frame_id: str,
        reason: str,
    ) -> None:
        record.delivered_count += 1
        record.valid = not reason
        record.reason = reason or "valid"
        record.valid_count += int(not reason)
        record.invalid_count += int(bool(reason))
        if record.first_sensor_stamp_ns is None:
            record.first_sensor_stamp_ns = sensor_stamp_ns
        record.last_sensor_stamp_ns = sensor_stamp_ns
        record.last_receive_time_ns = receive_time_ns
        record.frame_id = frame_id

    def _camera_info_consistent(
        self, image_stream: Stream, info_stream: Stream
    ) -> bool | None:
        image_record = self._records[image_stream]
        info_record = self._records[info_stream]
        image = image_record.observation
        info = info_record.observation
        if not isinstance(image, ImageObservation) or not isinstance(
            info, CameraInfoObservation
        ):
            return None
        return (
            image_record.valid
            and info_record.valid
            and image.width == info.width
            and image.height == info.height
            and image.frame_id == info.frame_id
        )
