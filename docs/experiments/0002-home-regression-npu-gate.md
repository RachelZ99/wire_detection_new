# Home regression NPU gate: evidence availability record

- Date: 2026-08-26
- Detection profile: `dcw2-home-640x360-v1`
- Rule version: `training-free-thin-line-v1`
- Current gate state: `EVIDENCE_INCOMPLETE`
- Ticket 8 state: closed; do not begin NPU model work

## Decision

No terminal NPU decision can be recorded from this checkout. The repository
contains the complete, versioned held-out scene plan and executable event-level
gate, but it contains none of the external home rosbag/result artifacts. In
particular, the available experiment notes describe only the original scene at
roughly 0.14 m/s; they do not provide the required independent 0.3 m/s straight
and turning recordings or the required stream/ground/NPU failure injections.

The absence of evidence does not mean the rule path passes, and it does not
identify a measured RGB failure class that would justify NPU work. Ticket 8
therefore remains closed until the suite produces either:

1. `RULE_PATH_PASSES`, in which case NPU segmentation is unnecessary for the
   agreed home gate; or
2. `NPU_REQUIRED` with concrete held-out RGB failure classes, in which case
   ticket 8 is limited to those classes.

Geometry, health, timing, profile, or resource failures produce
`NON_NPU_FAILURE` and do not justify cable segmentation.

## Scope

This record and all future reports from this suite are home feasibility
evidence, not factory validation. Factory data and factory acceptance remain
mandatory before any industrial safety claim.
