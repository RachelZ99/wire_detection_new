# Unified obstacle-response pre-integration contract

Status: local pre-integration only; not whole-robot acceptance.

## Boundary

The adapter subscribes to exactly these operational inputs:

| Topic | Type | QoS | Meaning |
| --- | --- | --- | --- |
| `/low_profile_hazard_perception/confirmed_hazards` | `sensor_msgs/msg/PointCloud2` | `RELIABLE`, `TRANSIENT_LOCAL`, `KEEP_LAST(1)` | Complete currently retained confirmed-hazard snapshot |
| `/low_profile_hazard_perception/health` | `diagnostic_msgs/msg/DiagnosticArray` | `RELIABLE`, `TRANSIENT_LOCAL`, `KEEP_LAST(1)` | 200 ms top-level perception heartbeat and profile binding |

The adapter has no `/cmd_vel`, slowdown, stop, emergency-stop, or replanning
publisher. It invokes an injected `UnifiedObstacleResponsePort` with two calls:

- `update_source_status(ResponseSourceStatus)` communicates source availability,
  top-level health, generation, and whether a fresh snapshot is awaited;
- `replace_confirmed_hazards(ConfirmedHazardSnapshot)` supplies the complete
  operational `odom` snapshot, including an explicit empty snapshot.

These calls are the local adapter seam, not a claim about the robot's existing
API. The real binding is blocked until the robot owner supplies the actual
message/service/action/plugin endpoint, QoS or call semantics, acknowledgements,
response timestamps, and failure behavior. The binding must call the unified
obstacle-response layer and must not control the chassis directly.

## Confirmed cloud contract

- `header.frame_id` is exactly `odom`.
- `header.stamp` is observation time, not publication or heartbeat time. For a
  non-empty snapshot it equals the oldest per-point observation stamp.
- Every point supplies `x/y/z`, `observation_stamp_sec`,
  `observation_stamp_nanosec`, `cloud_group_index`, and `hazard_track_id`.
  Points in one cloud group have one observation stamp and one track ID.
- A non-empty message replaces the complete retained snapshot. No repeated
  publication is required while that snapshot is unchanged; topic silence does
  not mean that a hazard disappeared.
- A zero-width message is an explicit empty snapshot. It may clear only under
  the configured state policy. Missing, stale, invalid, or degraded input is
  never converted to empty space.
- Perception retains confirmed hazards while they enter the observation blind
  zone. Downstream must not expire them merely because their cloud is not
  repeated.
- `hazard_track_id` is stable only within a perception process generation. A
  health-liveness loss/recovery or a regressing heartbeat identifies a possible
  restart; the bridge increments its generation and awaits the new snapshot
  before considering track identity current.

## Health and age contract

The operational status is the diagnostic named
`low_profile_hazard_perception/input_health`. The adapter allows only:

- top-level `HEALTHY`, `DEGRADED`, or `INVALID`;
- `profile.id`, `profile.binding_state`, `profile.maximum_speed_mps`, and
  `profile.latest_observed_speed_mps`;
- local steady-clock receive time for the 600 ms health-liveness timeout;
- cloud observation stamps and the local ROS clock for the 2.5 s provisional
  maximum observation age.

Health `header.stamp` is the publication heartbeat; it is not a hazard
observation stamp. Health liveness uses local receive time so a sensor-clock
pause cannot masquerade as a live publisher. The current profile must be
`dcw2-home-640x360-v1`, bound, and at or below 0.3 m/s.

Provider names, `resource.npu_state`, candidate diagnostics, evidence masks,
RGB masks, color thresholds, and provider-specific reasons are not parsed. In
the current rule profile, `disabled_rule_profile` is not an NPU failure. A
future NPU provider uses the same two topics and types; downstream reacts only
to its resulting top-level health and age. Geometry-confirmed output may still
be forwarded while top-level health is `DEGRADED`, subject to final safety
approval of ADR-0010.

## Provisional state behavior and blockers

| Input state | Non-empty confirmed snapshot | Explicit empty | Existing hazards |
| --- | --- | --- | --- |
| fresh `HEALTHY`, profile valid | forward | forward | replace |
| fresh `DEGRADED`, profile valid | forward as degraded | ignore | preserve until replacement |
| `INVALID`, health stale/missing | ignore | ignore | preserve |
| profile/speed/age invalid | ignore | ignore | preserve |

This table does not select a chassis response. Product/safety approval is still
required for the provisional timeouts, degraded admission/clearing, unified
layer retention, and the final state-to-`stop`/`avoid`/`reject` mapping.

## Isolated and physical verification

`audit_obstacle_response_trial` reads a versioned event log and derives
observation, confirmation, unified-receive and response-start times; command and
smoothed-command samples; wheel and fused odometry; detection distance; braking
start and stopping positions; stopping distance; hazard track; health
transitions; and final `stop`/`avoid`/`reject` outcome. Input SHA-256 is included
in the result. Fault injection defaults to mocks, offline replay, or an isolated
ROS domain.

The physical matrix is
`config/obstacle_response_physical_trial_manifest_v1.json`. Every case remains
`PLANNED_NO_EVIDENCE`. No remote robot connection, motion command, online-node
stop, or online ROS-graph fault injection is authorized by this work.
