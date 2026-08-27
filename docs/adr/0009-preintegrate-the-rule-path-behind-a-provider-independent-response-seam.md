---
status: accepted
---

# Pre-integrate the rule path behind a provider-independent response seam

Ticket 9 may proceed as local pre-integration while the home regression NPU
gate remains `EVIDENCE_INCOMPLETE`. The current detection profile is fixed to
`dcw2-home-640x360-v1` and its internal RGB provider is
`training_free_thin_line`. This is not a `RULE_PATH_PASSES` decision. Ticket 8
remains deferred; this work does not add a model, RKNN, or an NPU provider.

The downstream seam consumes only the confirmed `odom` PointCloud2 snapshot,
top-level health, profile binding, observation age, health liveness, and
`hazard_track_id`. Provider names, candidate diagnostics, masks, and internal
failure details are outside the seam. Replacing the internal RGB provider must
not change Ticket 9 topics, message types, or state-machine inputs.

No callable unified obstacle-response interface exists in this repository or
the available legacy project. The local implementation therefore defines an
injected port and a recording test double without inventing a production ROS
endpoint. Binding that port to the real robot remains blocked on the robot
owner supplying and approving the actual message, service, action, or plugin
contract. Ticket 9 whole-robot acceptance remains blocked on Ticket 7's
terminal decision and the required physical evidence.
