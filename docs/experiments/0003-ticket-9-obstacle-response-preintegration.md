# Ticket 9 obstacle-response pre-integration record

- Date: 2026-08-27
- Detection profile: `dcw2-home-640x360-v1`
- Current provider: `training_free_thin_line`
- Ticket 7 gate: `EVIDENCE_INCOMPLETE`
- Ticket 8: deferred; no NPU work performed
- Whole-robot evidence: `NOT_RUN`
- Ticket 9 release/physical acceptance: `BLOCKED`

## Local result

The provider-independent confirmed-cloud/health bridge, injected unified
response port, recording test double, isolated ROS graph test, and auditable
trial result tool are implemented. The available legacy project was inspected;
it contains only a demonstration `STOP/GO` signal and proposed Nav2/costmap
configuration, not the robot's existing unified obstacle-response API. Neither
was imported.

## Isolated ROS 2 Humble validation

The repository CI script was run on an Ubuntu 22.04 ARM64 VMware guest with
ROS 2 Humble. The test snapshot lived under `/tmp`; the guest's checkout was
left unchanged. Discovery was restricted with `ROS_DOMAIN_ID=97` and
`ROS_LOCALHOST_ONLY=1`, and no robot endpoint or motion-command publisher was
used.

The final `scripts/ros2_humble_ci.sh` run completed `colcon build`, collected
117 pytest cases, and reported `117 tests, 0 errors, 0 failures, 0 skipped`.
This includes the health, geometric-hazard, and unified-obstacle-response ROS
graph tests. The guest did not have the declared `python3-jsonschema` runtime
dependency installed system-wide, so `jsonschema` was installed only into the
temporary test directory for this run.

## Missing external contract

The robot owner must provide the real unified response endpoint and message,
service, action, or plugin types; endpoint QoS; snapshot/clearing semantics;
acknowledgement and response-start observability; restart behavior; and failure
mapping. Until then the bridge can be tested only against its recording double.

## Evidence status

No 0.3 m/s straight/turning, multi-angle, acceleration/braking, observation
blind-zone, fault-injection, repeated cable/power-strip, or eight-hour physical
evidence was generated. The versioned manifest is a plan, not a result. Ticket
7 must reach a terminal decision and the proposed safety policy must be
approved before whole-robot acceptance can be claimed.
