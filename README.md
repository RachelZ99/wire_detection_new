# Low-Profile Hazard Perception

This repository is the new asynchronous RGB-D perception project described in
the PRD and ADRs. Ticket 2 adds a depth-only strong-geometry path: it observes
the floor, aligns each depth observation at its own timestamp in `odom`, and
publishes only twice-confirmed low-profile hazards.

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
floor truth.

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

Points at least 15 mm above that observed floor enter the strong-geometry path
only when they have robust local metric support and physical span. Invalid
reflective-floor depth is ignored as evidence. Every candidate is transformed
with an interpolated odom pose at the depth sensor stamp; two observations must
associate within 80 mm and 350 ms before any operational output is published.
The resulting `sensor_msgs/PointCloud2` uses frame `odom` and the confirming
observation's sensor stamp. The node publishes no slowdown, stop, or replanning
command, and it exposes no candidate cloud on the operational topic.
