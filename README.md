# Low-Profile Hazard Perception

This repository is the new asynchronous RGB-D perception project described in
the PRD and ADRs. Independent depth and training-free RGB paths observe the
floor, align each observation at its own timestamp in `odom`, publish only
twice-confirmed low-profile hazards, and conservatively retain them through the
measured observation blind zone and sensing degradation.

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

The training-free RGB cable contracts and synthetic positive/negative replay
set can be checked without ROS with:

```bash
PYTHONPATH=low_profile_hazard_perception \
  python3 -m unittest discover \
  -s low_profile_hazard_perception/test -p 'test_rgb_cable_*.py'
```

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

Run the ticket-4 acceptance on the recorded pale-cable bag and an independent
white-cable scene. Scene-specific odom annotations must accompany the external
bags; the repository does not invent their coordinates:

```bash
ros2 run low_profile_hazard_perception replay_rgb_cable \
  /path/to/wire_rgbd_strip_01 --repeat 2 \
  --expected-cable-center ODOM_X ODOM_Y \
  --output reference-pale-cable.json

ros2 run low_profile_hazard_perception replay_rgb_cable \
  /path/to/white_cable_scene --repeat 2 \
  --expected-cable-center ODOM_X ODOM_Y \
  --output reference-white-cable.json
```

The cable replay requires the formal `training_free_thin_line` provider, at
least two processed RGB observations, a cable confirmation, and cable-evidence
points with bounded confirmation spread and physical span in the operational
`odom` cloud. It explicitly rejects the diagnostic pink comparison as an
operational provider.

Named negative scenes are listed in
`low_profile_hazard_perception/config/cable_negative_replay_manifest.yaml`.
For each external scene bag, pass its measured odom annotation, for example:

```bash
ros2 run low_profile_hazard_perception replay_rgb_cable \
  /path/to/long_shadow_scene --repeat 2 \
  --negative-only \
  --negative-cable-region long_shadow MIN_X MAX_X MIN_Y MAX_Y
```

The repository contains no ROS bags for these scenes, so real-scene replay
results cannot be claimed from this checkout; the synthetic fixtures exercise
the same category gates deterministically.

The geometric replay additionally requires at least one non-empty, meaningfully stamped
`odom` point cloud, deterministic point bytes and timestamps across runs, and
an observed camera-to-floor distance in the measured 0.20–0.25 m range. The
nominal 0.15 m TF value is reported as a consistency comparison, not used as
floor truth. It also rejects outputs without robust height/physical span,
cross-observation spread above 25 mm, or a trail-like cloud extent. The bag
must continue supplying valid floor depth after the hazard enters the
observation blind zone; the replay then verifies that the retained cloud is
not cleared before the configured two-second conservative interval.

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
reflective-floor depth is excluded from strong geometry and emitted separately
only when it forms bounded support-only invalid-depth evidence. Every candidate
is transformed with an interpolated odom pose at the depth sensor stamp; two
observations must associate within 80 mm and 350 ms before any operational
output is published.
Odom brackets wider than 100 ms or containing a 0.25 m/45° discontinuity cannot
support alignment or confirmation.
The resulting `sensor_msgs/PointCloud2` uses frame `odom` and the confirming
observation's sensor stamp. The node publishes no slowdown, stop, or replanning
command, and it exposes no candidate cloud on the operational topic. Alongside
standard `x/y/z`, each operational point carries the confirmation's
`confirmation_spread` and an `evidence_mask` (`1` for strong geometry, `2` for
RGB cable shape, `4` for weak height, and `8` for continuous invalid depth) so
replay can reject every misaligned event and identify the evidence carried by
each retained shape.

The internal field remains named `sensor_stamp_ns` intentionally: the current
DCW2 profile has `use_hardware_time: false`, and the repository has not yet
proved clock offset/drift compensation needed to call this a true capture time.
Host receipt/callback time is never substituted for it.

## Training-free RGB cable output

The RGB path runs on the native delivered image without erosion or resizing.
It scores local paired-edge contrast at multiple 2--6 px scales, groups
continuous thin structures, and gates them by pixel length, apparent-width
range and consistency, curve continuity, and projected physical span. It does
not encode pink, white, or any other cable color, and it has no legacy
5000-pixel component-area gate.

Only cells supported by the accepted observed-ground snapshot selected for the
RGB sensor stamp can generate RGB candidates. Accepted snapshots are held in a
small timestamp-ordered cache; an RGB frame waits as `ground:awaiting_depth`
until the depth stream has advanced past it, then deterministically selects the
nearest snapshot (preferring the earlier stamp on a tie). Nearby floor support
conservatively bridges the narrow invalid depth stripe caused by a cable, while
observed raised cells exclude hanging wires, table/tripod legs, and background
structure. Every accepted RGB pixel is positioned by ray intersection with the
selected observed ground plane, including cable pixels whose depth is invalid.
A ground model more than 500 ms from the RGB sensor stamp blocks new cable
evidence as `ground:stale`.

Each RGB observation uses its own sensor stamp for odom interpolation and enters
the same two-observation tracker, retained set, and transient-local operational
point cloud as strong geometry. Health exposes provider, candidate/confirmation
counts, processing/drop counts, ground-ray projection, and
`cable.rgb_depth_synchronizer=disabled`. A bounded, timestamp-ordered RGB
reorder queue holds frames only until the depth event-time watermark reaches
them; it does not form approximate-time image pairs.

The optional pink demo profile is disabled by default. When explicitly enabled,
it publishes only `cable.diagnostic_pink_pixel_count`; health also states
`cable.diagnostic_pink_operational=false`, and the count never reaches the
tracker or operational cloud.

The ROS-independent negative replay set represents empty reflective floor,
long shadows, one-pixel floor seams, hanging/background wires, table/tripod
legs, and cable reflections. The positive set covers pale and white cables at
2--5 px, including missing depth along the cable pixels. These are home
feasibility fixtures, not factory validation.

## Asynchronous mixed-evidence fusion

Depth frames keep strong protrusions, weak height, and invalid depth as separate
evidence sources. Weak height begins at the greater of 6 mm or three estimated
robust ground-noise deviations and remains below the 15 mm strong geometry
gate. Invalid depth is accepted as support only when it forms a narrow,
continuous region enclosed by valid observed-floor depth; scattered or broad
holes do not become candidates.

The `odom` tracker accepts cross-stream observations in sensor-stamp order or
arrival order and converges on the same retained hazard. Two strong geometry
observations, two RGB cable observations at or above confidence 0.75, or one
strong geometry plus one high-confidence RGB observation can confirm. Weak
height and invalid depth can strengthen a spatially matching cable or
strong-geometry track, but never count as a confirming observation;
low-confidence RGB likewise cannot confirm. Two candidate
components from the same sensor stamp count as one observation. Odom evidence
coordinates are canonicalized at 0.1 mm, below meaningful detection precision,
to prevent insignificant floor-fit floating-point jitter from changing replay
bytes.

Every generated candidate is published separately on
`/low_profile_hazard_perception/candidate_diagnostics` as a
`diagnostic_msgs/DiagnosticArray`. Each status carries its own sensor stamp,
`odom` centroid, aggregate evidence sources, observed-ground acceptance and
quality metrics, confidence, and a stable decision reason such as
`SUPPORT_ONLY`, `LOW_CONFIDENCE_RGB`, `WAITING_FOR_CONFIRMATION`, or
`CONFIRMED_MIXED_EVIDENCE`. Depth components rejected for insufficient support
or span, excessive invalid-region width, or missing valid-depth enclosure are
also reported with stable `REJECTED_*` reasons. This debug topic is not the
operational navigation interface; only retained confirmed hazards appear on
`/low_profile_hazard_perception/confirmed_hazards`.

## Bound detection profile and resource budget

The formal rule path loads
`low_profile_hazard_perception/config/detection_profile_dcw2_home_640x360_v1.json`
at startup. Its schema version, profile ID, SHA-256 fingerprint, delivered
640x360@10 Hz stream and encodings, validated 8--12 Hz delivery range, measured
0.20--0.25 m height and 1.5--4.0 degree downward-pitch installation range,
footprint, thresholds, retention, rule/model versions, and 0.3 m/s maximum are
published under `profile.*` in health. The ROS parameter file remains the launch
surface, but every detection parameter must exactly match the versioned profile
or the node refuses to start. A delivered 640x400 image, an observed 0.60 m
installation, a materially different delivered frame rate or mount pitch, or
configured/odom-observed speed above 0.3 m/s produces an explicit profile
mismatch and blocks new confirmation.

All middleware image histories and input work queues are latest-only. The depth
work slot is also latest-only; the separate RGB event-time reorder buffer stays
bounded and drops its oldest observation at capacity. Health publishes current
pending counts and cumulative input, depth, RGB, and middleware drop counts.

`stage.depth_geometry.*`, `stage.rgb_cable.*`, and `stage.perception.*` expose a
fixed-size measurement window with processing wall time, process CPU time,
queue wait, sensor-message age, P95, and average CPU-core use. `resource.*`
exposes process CPU cores, current/peak RSS, RSS growth, retained sample count,
and NPU state. The rule profile reports the NPU as
`disabled_rule_profile`; it does not imply an NPU failure. After a minimum
sample count, exceeding the provisional 80 ms P95 or one-core depth budget
degrades health with a `budget:*` reason.

Run the complete two-hour resource acceptance against the external reference
bag on the RK3588 target:

```bash
ros2 run low_profile_hazard_perception soak_detection_profile \
  /path/to/wire_rgbd_strip_01 \
  --duration-seconds 7200 \
  --output dcw2-home-640x360-v1-soak.json
```

The command loops the bag, checks that the exact profile ID and fingerprint
remain bound, and writes measured P95 processing time, average depth CPU cores,
memory growth, maximum pending input work and RGB reorder depth, frame drops,
NPU state, sample counts, and machine-readable failure reasons. It also fails
if the perception launch exits, health becomes stale, or stage processing stops
progressing. Short runs are deliberately
reported as `soak_duration_incomplete`; they cannot be presented as the
two-hour acceptance. The bag and RK3588 runtime are external artifacts, so this
checkout supplies the reproducible measurement gate but does not claim that
the hardware soak has already passed.

## Home event regression and NPU gate

The versioned home scene plan is
`low_profile_hazard_perception/config/home_regression_manifest_v1.json`. Its
held-out acceptance split covers cable color/material and layout variation,
power strips/plugs, the specified difficult negatives, all reporting strata,
and actual 0.3 m/s straight and turning evidence. Tuning and acceptance videos
cannot share a scene group or bag ID.

After black-box replay has produced one normalized result JSON per scene, run:

```bash
ros2 run low_profile_hazard_perception audit_home_regression \
  --results-directory /path/to/home-regression-results \
  --output home-regression-report.json \
  --decision-record home-regression-npu-decision.md
```

The result contract is
`low_profile_hazard_perception/config/home_regression_scene_result_schema_v1.json`.
The audit reports event recall, persistent false events per hour, confirmed
detection distance, confirmation latency, health failures, resource use, and
the same metrics stratified by distance, cable appearance, floor, light, robot
motion, and depth validity. It verifies the bound profile/rule identity,
two-pass determinism, bag fingerprints, actual evaluated duration, and actual
odom-observed speed.

`RULE_PATH_PASSES` keeps ticket 8 closed. `NPU_REQUIRED` is emitted only when a
complete held-out suite identifies one or more configured RGB cable failure
classes. Missing evidence yields `EVIDENCE_INCOMPLETE`; geometry, health,
timing, or resource failures yield `NON_NPU_FAILURE`, and neither state is
misrepresented as a reason to train a cable model. See
`docs/home-regression-suite.md` and the current evidence record in
`docs/experiments/0002-home-regression-npu-gate.md`.

All home reports are feasibility evidence, not factory validation.

## Conservative degradation and retention

Unconfirmed candidates expire after 500 ms and can only confirm inside the
separate 350 ms confirmation window. Confirmed hazards use an independently
configured retention floor of 2000 ms; configurations below two seconds are
rejected. The operational topic is transient-local and represents the current
retained set. A non-empty set is republished only when its observations change,
and one stamped empty cloud clears it after safe expiry, keeping deterministic
replay independent of timer frequency. Every point carries its own source
seconds/nanoseconds; the cloud header conservatively uses the oldest source
stamp in the retained set.

Ground rejection, missing/stale/invalid CameraInfo or TF, and missing, stale,
disordered, gapped, or discontinuous odom block new cross-frame confirmation
and publish a machine-readable diagnostic reason. These states clear candidate
accumulation but never clear a retained confirmed hazard. Confirmed expiry is
suspended while health cannot support a safe interpretation; a spatially
consistent pair of observations after recovery refreshes the same retained
hazard rather than duplicating or teleporting it. The first recovery observation
is diagnostic-only and cannot replace the operational footprint.

The configured retention and current behavior are visible as
`geometry.candidate_retention_ms`, `geometry.confirmed_retention_ms`,
`geometry.active_retained_hazard_count`, `geometry.degradation_reason`, and
`geometry.output_durability` in the health diagnostic.
