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
