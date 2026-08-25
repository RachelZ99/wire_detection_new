# Low-Profile Hazard Perception

This repository is the new asynchronous RGB-D perception project described in
the PRD and ADRs. The current depth-only path observes the floor, aligns each
depth observation at its own timestamp in `odom`, publishes only twice-confirmed
low-profile hazards, and conservatively retains them through the measured
observation blind zone and sensing degradation.

## Supported environment

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10 (the Humble system Python)
- `ros-humble-ros-base`, `python3-colcon-common-extensions`, and the package
  dependencies resolved by `rosdep`

Build and run the complete test suite with:

```bash
./scripts/ros2_humble_ci.sh
```

The equivalent explicit commands are:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src --rosdistro humble -y
colcon build --symlink-install --packages-select low_profile_hazard_perception
source install/setup.bash
colcon test --packages-select low_profile_hazard_perception
colcon test-result --verbose
```

The ROS-independent health contract can also be checked without ROS:

```bash
PYTHONPATH=low_profile_hazard_perception \
  python3 -m unittest discover \
  -s low_profile_hazard_perception/test -p 'test_health_monitor.py'
```

The geometry and temporal contracts can be checked the same way with
`test_ground_geometry.py`, `test_geometric_pipeline.py`, and
`test_temporal_alignment.py`.

## Deterministic reference replay

The reference bag is an external test artifact named `wire_rgbd_strip_01`; it
is intentionally not copied into Git. After building and sourcing the workspace,
run it at real time twice:

```bash
ros2 run low_profile_hazard_perception replay_rgbd_health \
  /path/to/wire_rgbd_strip_01 \
  --repeat 2 \
  --output reference-health.json
```

Run the ticket-2 operational replay separately:

```bash
ros2 run low_profile_hazard_perception replay_geometric_hazard \
  /path/to/wire_rgbd_strip_01 \
  --repeat 2 \
  --output reference-geometric-hazard.json
```

This replay additionally requires at least one non-empty, meaningfully stamped
`odom` point cloud, deterministic point bytes and timestamps across runs, and
an observed camera-to-floor distance in the measured 0.20–0.25 m range. The
nominal 0.15 m TF value is reported as a consistency comparison, not used as
floor truth. It also rejects outputs without robust height/physical span,
cross-observation spread above 25 mm, or a trail-like cloud extent.

For event-level acceptance, add the measured `odom` annotations for the bag:

```bash
ros2 run low_profile_hazard_perception replay_geometric_hazard \
  /path/to/wire_rgbd_strip_01 --repeat 2 \
  --expected-power-strip-center ODOM_X ODOM_Y \
  --reflective-floor-region MIN_X MAX_X MIN_Y MAX_Y
```

The annotated form requires a strong confirmed cloud at the power-strip region
and rejects two or more confirmed clouds in each reflective-floor negative
region. The repository does not invent these coordinates: they must come from
the reference-scene annotation paired with the external bag.

The command starts a clean input-health node for each pass, replays with ROS
simulation time, and fails if delivered input counts or canonical health fields
differ. It also fails unless the final state is `HEALTHY`, and fails if any
transient `INVALID` state occurred. Canonical comparison excludes host-dependent
age and processing-latency values; the JSON output retains their per-run ranges
and the complete sequence of health transitions.

The replayed inputs are independent subscriptions; there is no RGB-depth
`ApproximateTime` gate or synchronized image-pair callback. The subscribed
topics are:

```text
/camera_1/color/image_raw
/camera_1/color/camera_info
/camera_1/depth/image_raw
/camera_1/depth/camera_info
/odom
/tf
/tf_static
```

## Health output

Launch the node and inspect its latched diagnostic output:

```bash
ros2 launch low_profile_hazard_perception input_health.launch.py
ros2 topic echo /low_profile_hazard_perception/health
```

The `diagnostic_msgs/DiagnosticArray` reports:

- delivered 640×360 profiles, `rgb8`/`16UC1` encodings, frame IDs, sensor-stamp
  rates, delivered/valid/invalid counts, and CameraInfo consistency;
- sensor-stamp age from the ROS clock, receive age from the host steady clock,
  and receive-to-processing-complete latency as separate measurements;
- missing streams, sensor-stale streams, receive-stale streams, capacity-one
  application queue drops, and middleware transport drops as separate fields;
- `DEGRADED` for absent/stale/lost inputs and `INVALID` for malformed image,
  CameraInfo, odom, or TF contracts.

`queue_drops` comes from explicit drop-oldest application queues and
`transport_drops` comes only from the middleware `message_lost` event; neither
is inferred from rate or age. Image middleware histories and processing work
are both bounded to the latest sample, so overload drops old work instead of
accumulating decision latency. No `sensor_msgs/PointCloud2`, slowdown, stop, or
replanning output is created by the standalone `input_health` node.

## Strong geometric hazard output

Launch the geometry path and inspect its two separate outputs:

```bash
ros2 launch low_profile_hazard_perception geometric_hazard.launch.py
ros2 topic echo /low_profile_hazard_perception/health
ros2 topic echo /low_profile_hazard_perception/confirmed_hazards
```

Each valid depth frame is sparsely deprojected and fitted with a deterministic,
direction-constrained robust observed-ground model. Health exposes its support,
inlier ratio, median/P90 metric residual, spatial coverage, temporal
consistency, measured camera height, and disagreement with nominal TF.
Abruptly inconsistent ground models are rejected rather than smoothed into an
accepted model. The observed normal and camera height correct the nominal
camera-to-base tilt and height before points are carried into `odom`.

Points at least 15 mm above that observed floor enter the strong-geometry path
only when they have robust local metric support and physical span. Invalid
reflective-floor depth is ignored as evidence. Every candidate is transformed
with an interpolated odom pose at the depth sensor stamp; two observations must
associate within 80 mm and 350 ms before any operational output is published.
Odom brackets wider than 100 ms or containing a 0.25 m/45° discontinuity cannot
support alignment or confirmation.
The resulting `sensor_msgs/PointCloud2` uses frame `odom` and the confirming
observation's sensor stamp. The node publishes no slowdown, stop, or replanning
command, and it exposes no candidate cloud on the operational topic. Alongside
standard `x/y/z`, each operational point carries the confirmation's
`confirmation_spread` so replay can reject every misaligned event rather than
only inspecting the last health snapshot.

The internal field remains named `sensor_stamp_ns` intentionally: the current
DCW2 profile has `use_hardware_time: false`, and the repository has not yet
proved clock offset/drift compensation needed to call this a true capture time.
Host receipt/callback time is never substituted for it.

## Conservative degradation and retention

Unconfirmed candidates expire after 500 ms and can only confirm inside the
separate 350 ms confirmation window. Confirmed hazards use an independently
configured retention floor of 2000 ms; configurations below two seconds are
rejected. The operational topic is transient-local and represents the current
retained set. A non-empty set is republished only when its observations change,
and one stamped empty cloud clears it after safe expiry, keeping deterministic
replay independent of timer frequency.

Ground rejection, missing/stale/invalid CameraInfo or TF, and missing, stale,
disordered, gapped, or discontinuous odom block new cross-frame confirmation
and publish a machine-readable diagnostic reason. These states clear candidate
accumulation but never clear a retained confirmed hazard. Confirmed expiry is
suspended while health cannot support a safe interpretation; a spatially
consistent observation after recovery refreshes the same retained hazard rather
than duplicating or teleporting it.

The configured retention and current behavior are visible as
`geometry.candidate_retention_ms`, `geometry.confirmed_retention_ms`,
`geometry.active_retained_hazard_count`, `geometry.degradation_reason`, and
`geometry.output_durability` in the health diagnostic.
