# Home regression suite

The home regression suite is the event-level acceptance seam for the bound
`dcw2-home-640x360-v1` rule path. It consumes results from complete black-box
RGB-D/CameraInfo/TF/odom replay runs; it does not call detector internals, form
synchronized RGB-depth pairs, or reuse the legacy color-specific node.

The versioned scene plan is
`low_profile_hazard_perception/config/home_regression_manifest_v1.json`. It
keeps tuning videos and held-out acceptance videos in different
`scene_group_id` and `bag_id` groups. The held-out plan covers:

- pink rubber, white PVC, black rubber, and gray braided cables;
- straight, curved, crossing, partially occluded, and raised layouts;
- power strips and plugs through the independent geometry path;
- reflections, shadows, floor seams, scratches, tape, furniture legs, mat
  edges, and empty floor;
- near/middle/far distances, three floor/lighting conditions, valid and
  cable-invalid depth, stationary motion, and 0.3 m/s straight and turning
  recordings.

The bag files and their result JSON files are external evidence artifacts and
are not invented or checked into this repository. Each result must conform to
`home_regression_scene_result_schema_v1.json` in the same config directory.
In particular, a result records the bag SHA-256, two-pass determinism, actual
evaluated duration, maximum speed observed from odom, per-event detection
distance and confirmation latency, persistent false events, health failures,
and worst resource use.

Event metrics come from the operational interface:

- an event is detected only when an annotated hazard overlaps a confirmed,
  stamped `odom` point cloud;
- confirmation latency is the interval between the first and confirming
  independent source stamps, not callback time;
- confirmed detection distance is robot-to-hazard distance at the confirming
  source stamp after odom interpolation;
- one retained hazard is one event; republished frames do not increase recall;
- a false event is a confirmed hazard in an annotated negative region, not a
  candidate mask pixel.

Run the audit after placing one result file per manifest scene in a directory:

```bash
ros2 run low_profile_hazard_perception audit_home_regression \
  --results-directory /path/to/home-regression-results \
  --output home-regression-report.json \
  --decision-record home-regression-npu-decision.md
```

Pass a non-default manifest as the first positional argument. The command
writes SHA-256 fingerprints for the manifest and every consumed result. Missing
or mismatched results produce `EVIDENCE_INCOMPLETE`; they can never pass the
rule path or select NPU work.

The terminal decisions are:

- `RULE_PATH_PASSES`: held-out event/health/resource gates pass; ticket 8 stays
  closed.
- `NPU_REQUIRED`: the complete gate fails only in named RGB cable failure
  classes listed by the manifest; ticket 8 runs against those measured classes.
- `NON_NPU_FAILURE`: geometry, health, timing, resource, or an unapproved
  failure class fails; cable segmentation is not treated as a remedy.
- `EVIDENCE_INCOMPLETE`: required scenes, actual 0.3 m/s straight/turning
  evidence, identity, determinism, or result fields are missing.

Every generated report is labeled home feasibility evidence. It is not factory
validation and cannot support an industrial safety claim.
